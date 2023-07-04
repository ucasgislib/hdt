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
import time,math
import os
from io import BytesIO
from matplotlib import pyplot as plt
import geopandas as gpd
import re
import psycopg2

from shapely.geometry import Point,Polygon,LineString
from sqlalchemy import create_engine
from sqlalchemy import select,Table
from sqlalchemy import func,extract
from sqlalchemy.orm import aliased,registry

import datetime
import pytz

from pyproj import Proj
from geopy.distance import great_circle
from shapely.geometry  import MultiPoint,Polygon
from geopy.geocoders  import Nominatim



plt.rcParams['font.family'] = 'Times New Roman' # 设置字体样式
plt.rcParams['font.sans-serif'] = 'SimHei'

plt.rcParams['figure.figsize'] = (12, 6)     # 显示图像的最大范围
# plt.rcParams['figure.figsize'] = (8, 6)     # 显示图像的最大范围

plt.rcParams['font.size'] = 16 # 设置字体大小
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['savefig.dpi'] = 600

class PLOT_KDE(object):
    """docstring for GPS_KDE"""
    def __init__(self):
        self.x0,self.y0 = [12948949.901, 4877639.899]
        self.max_dist = 2000
        self.init_proj()

        self.host='10.120.16.20'  #服务器地址
        self.db_name='sx_2023'  
        self.t_name='car_all'
        
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

    def get_carID(self):

        self.cur.execute("SELECT distinct(cid) FROM %s "%self.t_name)
        cars = self.cur.fetchall()
        fw = open("../csv/car_ID.csv","w")
        for row in cars:
            fw.write(row[0]+"\n")
        fw.close()

    def init_proj(self):
        params = {'proj': 'merc', 'a': 6378137, 'b': 6378137, 'lat_ts': 0, 'lon_0': 0, 'x_0': 0, 'y_0': 0, 'k': 1.0, 'units': 'm', 'nadgrids': '@null'}
        self.proj3857 = Proj(params)

    def cal_Eu_dist(self,x1,y1,x2,y2):
        x,y = x2-x1, y2-y1
        tda = math.sqrt(x**2+y**2)
        # tda = x**2+y**2
        
        return tda

    def get_centermost_point(self,cluster):
        # print cluster
        cluster = [[float(c[1]),float(c[0])] for c in cluster]
        # print cluster
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        # print great_circle(point, centroid).m
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        
        return tuple(centermost_point)

    def cal_rog(self,usr_tslist):
        ts_len = len(usr_tslist)

        xy = self.get_centermost_point(usr_tslist)
        cx,cy = self.proj3857(float(xy[1]),float(xy[0]))
        ts_dist = 0
        for rec in usr_tslist:
            try:
                lon,lat = rec
                x,y = self.proj3857(lon,lat)
                dis = self.cal_Eu_dist(x,y,cx,cy)
                ts_dist+=(dis/1000.00)
            except:
                pass
        if ts_len !=0:
            rog = math.sqrt(ts_dist/ts_len)
        else:
            rog = 0

        return rog

    def stat(self,cid,trj):
        # ts,lons,lats = trj['ts'],trj['lon'],trj['lat']
        # func = lambda p:self.proj3857(p[0],p[1])
        # xys = np.array(list(map(func,np.c_[lons,lats])))
        # xs,ys = xys[:,0],xys[:,1]

        stays, dists = [],[]

        recs = trj.values
        cp = len(recs)

        timf,cid,lon0,lat0,tim_ = recs[0]
        x_,y_ = self.proj3857(lon0,lat0)

        usr_tslist = []
        for i,rec in enumerate(recs):
            if i == 0 or i == cp:
                continue
            timf,cid,lon,lat,tim = rec
            staytime = int(tim - tim_)
            x,y = self.proj3857(lon,lat)
            dist = np.sqrt((x-x_)**2+(y-y_)**2)/1000
            stays.append(staytime)
            dists.append(dist)
            timf_,cid_,lon_,lat_,tim_ = rec
            
            usr_tslist.append([lon,lat])

        rog = self.cal_rog(usr_tslist)

        # print(cp,np.mean(stays),np.mean(dists),rog)

        return cp,rog,stays,dists


    def run(self):

        car_info = pd.read_csv("../TrajData/shaanxi/车辆.csv")
        print(car_info.columns)
        result_dict = car_info.set_index('车辆编号')[['车辆类别','车辆状态','燃料类型']].to_dict()
        # print(result_dict['车辆类别'])


        #self.get_carID()
        cids = pd.read_csv("../csv/car_ID.csv",names=['cid'])['cid'].tolist()
        print(len(cids))


        # cp = 0
        # fw = open("../csv/car_ID_hazardous.csv",'w')
        # for cid in cids:
        #     # print(cid,result_dict["车辆类别"][cid])
        #     try:
        #         if result_dict["车辆类别"][cid] == "危险品" and result_dict["车辆状态"][cid] == "正常":
        #             fw.write(str(cid))
        #             cp+=1
        #         else:
        #             pass
        #     except:
        #         # cp+=1
        #         pass
        # print(cp)
        # fw.close()

        hdt_cars = []
        cp = 0
        for cid in cids:
            # print(cid,result_dict["车辆类别"][cid])
            try:
                if (result_dict["车辆类别"][cid] == "货运" or result_dict["车辆类别"][cid] == "危险品"):
                    if result_dict["车辆状态"][cid] == "正常" and result_dict["燃料类型"][cid] == "柴油":
                        hdt_cars.append(cid)
                else:
                    cp+=1
            except:
                # cp+=1
                pass
        print(len(hdt_cars),cp)


        t0 = time.time()
        cp_ = 0

        fw = open("../csv/hdt_rog2km.csv",'w')
        fw.write("cid,pnts,rog\n")
        for cid in hdt_cars:
            sql = "select * from %s where cid = '%s' "%(self.t_name,cid)
            self.cur.execute(sql)

            traj = self.cur.fetchall()

            trj = pd.DataFrame(traj, columns=['cid', 'tim', 'lon', 'lat', 'spe', 'speed', 'legend', 'height', 'direction', 'status', 'warn'])            
            sel_cols = ['tim','cid','lon','lat']
            trj = trj[sel_cols]
            trj['ts'] = trj['tim'].apply(lambda x: pd.Timestamp(x).timestamp())
            trj = trj.sort_values(by='ts')

            try:
                cp,rog,stays,dists = self.stat(cid,trj)
                if rog > 2:
                    fw.write(",".join(str(r) for r in [cid,cp,rog])+"\n")
            except:
                pass
            
            cp_+=1
            if cp_ % 100 ==0:
                t1 = time.time()
                print(cp_,len(hdt_cars),(t1-t0)/60)
        fw.close()
        

if __name__ == '__main__':
    PK = PLOT_KDE()
    PK.run()