#!/usr/bin/python
# coding=utf-8

from matplotlib import gridspec
from itertools import groupby
from pyproj import Proj


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import random
import time
import os
from io import BytesIO
from matplotlib import pyplot as plt
import geopandas as gpd
import re
import psycopg2

from shapely.geometry import Point,Polygon,LineString
from shapely import wkt
from sqlalchemy import create_engine
from sqlalchemy import select,Table
from sqlalchemy import func,extract
from sqlalchemy.orm import aliased,registry

import datetime
import pytz

from trace_segment import TraceProcess, keep_region, clip_traj, bp_detect

import warnings
warnings.filterwarnings("ignore")


class TRAJ(object):
    """docstring for GPS_KDE"""
    def __init__(self):
        self.x0,self.y0 = [12948949.901, 4877639.899]
        self.max_dist = 2000
        self.init_proj()

        self.host='10.120.16.20'  #服务器地址
        self.db_name='sx_2023'  
        self.t_name='car_all'
        self.engine = create_engine("postgresql://jpnms:wang@{}:5432/{}".format(self.host, self.db_name))
        
        # 连接到远程数据库
        self.conn = psycopg2.connect(
            host=self.host,
            database=self.db_name,
            user="jpnms",
            password="wang"
        )

        # 创建游标对象
        self.cur = self.conn.cursor()

    def __del__(self):
        self.cur.close()
        self.conn.close()


    def get_table(self,engine,host, t_name, db_name):

        mapper_registry = registry()
        return Table(t_name, mapper_registry.metadata, autoload_with=engine)

    def init_proj(self):
        params = {'proj': 'merc', 'a': 6378137, 'b': 6378137, 'lat_ts': 0, 'lon_0': 0, 'x_0': 0, 'y_0': 0, 'k': 1.0, 'units': 'm', 'nadgrids': '@null'}
        self.proj3857 = Proj(params)


    def p2str(self, data, geom_type):
        """

        @param data:
        @param geom_type:
        @return:
        """
        line = "%s (" % geom_type
        for x, y in zip(data.lng.astype(str).values, data.lat.astype(str).values):
            line += (x + " " + y + ",")
        line=line.strip(',')
        line += (')')



        return line


    def gps2od(self, data, epsg_target, speed_max, speed_min, duration, gap_max, dis_max):
        """

        @param data:
        @param region_map:
        @param epsg_target:
        @param speed_max:
        @param speed_min:
        @param duration:
        @param gap_max:
        @param dis_max:
        @return:
        """

        # data.tim = pd.to_datetime(data.tim, unit='s')
        # print(data)
        # exit()

        # 开始处理
        data["pid"] = data.index
        processer = TraceProcess(data, epsg_target=epsg_target)
        
        # 漂移去除
        drift_list = processer.drift_detect(speed_max=speed_max)
        length = processer.clean_point(drift_list)
        if length < 3:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        pp_move = np.array([], dtype=int)  # 静止段起始点
        pp_stay = np.array([], dtype=int)  # 移动段起始

        trace_keep = data.index.values
        points_bp = bp_detect(trace_keep)

        # 行程划分切割（轨迹点间隔、间距）
        points_pair = clip_traj(processer=processer, trace_keep=trace_keep, points_bp=points_bp, gap_max=gap_max,
                                dis_max=dis_max)
        
        # 对区域内轨迹做停留检测
        for pr in points_pair:
            if (pr[1] - pr[0]) > 2:
                stay, move, index = processer.stay_move_detect(clips=pr, speed_min=speed_min, dura_second=duration)
                pp_move = np.append(pp_move, move)
                pp_stay = np.append(pp_stay, stay)

        pp_move = pp_move.reshape(-1, 2)
        pp_stay = pp_stay.reshape(-1, 2)

        # 保留运动路程段表
        move_pd = pd.DataFrame(None, columns=['cid', 'sp', 'ep', 'stime', 'etime','slng','slat','elng','elat'])
        move_pd.cid = processer.trace.cid.iloc[pp_move[:, 0]].values
        move_pd.stime = processer.trace.time.iloc[pp_move[:, 0]].values
        move_pd.etime = processer.trace.time.iloc[pp_move[:, 1]].values
        move_pd.sp = processer.trace.pid.iloc[pp_move[:, 0]].values
        move_pd.ep = processer.trace.pid.iloc[pp_move[:, 1]].values
        move_pd.slng = processer.trace.lng.iloc[pp_move[:, 0]].values
        move_pd.slat = processer.trace.lat.iloc[pp_move[:, 0]].values
        move_pd.elng = processer.trace.lng.iloc[pp_move[:, 1]].values
        move_pd.elat = processer.trace.lat.iloc[pp_move[:, 1]].values

        # 运动line生成
        time_list = []  # 时间序列
        move_list = []  # line 序列

        for pairs in pp_move:
            gps_points = processer.trace.iloc[pairs[0]:pairs[1] + 1]
            trace_line = self.p2str(gps_points[['lng', 'lat']], geom_type="LINESTRING")  # LINESTRING MULTIPOINT
            time_list.append(gps_points.time.astype(str).to_list())
            move_list.append(trace_line)

        line_tmp = pd.DataFrame(None, columns=['cid', 'time', 'geom'])
        line_tmp['geom'] = move_list
        line_tmp["time"] = time_list
        line_tmp['cid'] = move_pd.cid

        # 静止点提取
        pids_list = []
        stay_list = []
        for pairs in pp_stay:
            gps_points = processer.trace.iloc[pairs[0]:pairs[1] + 1]
            rests_mupo = self.p2str(gps_points[['lng', 'lat']], geom_type="MULTIPOINT")
            pids_list.append(gps_points.pid.to_list())
            stay_list.append(rests_mupo)

        rest_tmp = pd.DataFrame(None, columns=['cid', 'pids', 'geom'])
        rest_tmp["geom"] = stay_list
        rest_tmp["cid"] = data.cid.iloc[-1]
        rest_tmp["pids"] = pids_list

        return move_pd, line_tmp, rest_tmp


    def run(self):
        cids = pd.read_csv("../csv/hdt_rog2km.csv")['cid'].tolist()

        cp = 0
        t0 = time.time()
        pd_ods = pd.DataFrame(None, columns=['cid', 'sp', 'ep', 'stime', 'etime','slng','slat','elng','elat'])

        for cid in cids:
            sql = "select * from %s where cid = '%s' "%(self.t_name,cid)
            self.cur.execute(sql)

            traj = self.cur.fetchall()
            trj = pd.DataFrame(traj, columns=['cid', 'time', 'lng', 'lat', 'spe', 'speed', 'legend', 'height', 'direction', 'status', 'warn'])
            try:
                move_pd, line_tmp, rest_tmp = self.gps2od(trj, epsg_target=900913, speed_max=40, speed_min=1, duration=1800, gap_max=3600,
                                                     dis_max=300000)

                move_pd.to_csv("../shp/trj_ODs/car_%s_ods.csv"%cid)

                pd_ods = pd.concat([pd_ods, move_pd])

                # trj['ts'] = trj['time'].apply(lambda x: pd.Timestamp(x).timestamp())
                # trj = trj.sort_values(by='ts')
                
                # trj['datetime_str'] = trj['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                # trj = trj.drop(columns=['time', 'spe', 'speed', 'legend', 'height', 'direction', 'status', 'warn'])
                # trj = trj.rename(columns={'datetime_str': 'datetime'})

                # gdf_pnts = gpd.GeoDataFrame(trj, geometry=gpd.points_from_xy(trj['lng'], trj['lat']))
                # gdf_line = gdf_pnts.groupby('cid')['geometry'].apply(lambda x: LineString(x.tolist()))

                # gdf_pnts.to_file("../shp/trj_ODs/car_%s_pnts.shp"%cid, driver='ESRI Shapefile')
                # gdf_line.to_file("../shp/trj_ODs/car_%s_lines.shp"%cid, driver='ESRI Shapefile')
                
                # line_tmp['geometry'] = line_tmp['geom'].apply(wkt.loads) 
                # rest_tmp['geometry'] = rest_tmp['geom'].apply(wkt.loads) 
                
                # gdf_moves = gpd.GeoDataFrame(line_tmp[['cid','geometry']], geometry='geometry')
                # gdf_rests = gpd.GeoDataFrame(rest_tmp[['cid','geometry']], geometry='geometry')

                # gdf_moves.to_file("../shp/trj_ODs/car_%s_moves.shp"%cid, driver='ESRI Shapefile')
                # gdf_rests.to_file("../shp/trj_ODs/car_%s_rests.shp"%cid, driver='ESRI Shapefile')

            except:
                pass

            cp+=1
            if cp % 1000 == 0:
                t1 = time.time()
                print(cp,len(cids),(t1-t0)/60)
        pd_ods.to_csv("../csv/ods.csv")

if __name__ == '__main__':
    PK = TRAJ()
    PK.run()