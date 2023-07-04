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
import geopandas as gpd


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
        pass
        # self.cur.close()
        # self.conn.close()

    def get_carID(self):

        self.cur.execute("SELECT distinct(cid) FROM %s "%self.t_name)
        cars = self.cur.fetchall()
        fw = open("../csv/car_ID.csv","w")
        for row in cars:
            fw.write(row[0]+"\n")
        fw.close()

    def get_signle_car(self,cid):
        sql = "select * from %s where cid = '%s' "%(self.t_name,cid)
        self.cur.execute(sql)

        traj = self.cur.fetchall()

        trj = pd.DataFrame(traj, columns=['cid', 'tim', 'lon', 'lat', 'spe', 'speed', 'legend', 'height', 'direction', 'status', 'warn'])
            
        sel_cols = ['tim','cid','lon','lat']
        trj = trj[sel_cols]
        trj['ts'] = trj['tim'].apply(lambda x: pd.Timestamp(x).timestamp())
        trj = trj.sort_values(by='ts')
        trj = trj[['ts','cid','lon','lat']]

        gdf_pnts = gpd.GeoDataFrame(trj, geometry=gpd.points_from_xy(trj['lon'], trj['lat']))
        gdf_line = gdf_pnts.groupby('cid')['geometry'].apply(lambda x: LineString(x.tolist()))

        gdf_pnts.to_file("../shp/cid_shp/car_%s_pnts.shp"%cid, driver='ESRI Shapefile')
        gdf_line.to_file("../shp/cid_shp/car_%s_line.shp"%cid, driver='ESRI Shapefile')

    def get_cars(self,cids):

        sql = "SELECT * FROM %s  WHERE cid IN (SELECT distinct(cid) FROM %s limit 20000);"%(self.t_name,self.t_name)
        # sql = "select * from %s where cid = '%s' "%(self.t_name,cid)
        print(sql)
        self.cur.execute(sql)

        traj = self.cur.fetchall()
        
        print('tim','cid','lon','lat')

        trj = pd.DataFrame(traj, columns=['cid', 'tim', 'lon', 'lat', 'spe', 'speed', 'legend', 'height', 'direction', 'status', 'warn'])
               
        sel_cols = ['tim','cid','lon','lat']
        trj = trj[sel_cols]
        trj['ts'] = trj['tim'].apply(lambda x: pd.Timestamp(x).timestamp())
        trj = trj.sort_values(by='ts')
        trj = trj[['ts','cid','lon','lat']]

        trj = trj.groupby("cid").filter(lambda x: len(x) >= 100)

        gdf_pnts = gpd.GeoDataFrame(trj, geometry=gpd.points_from_xy(trj['lon'], trj['lat']))

        gdf_line = gdf_pnts.groupby('cid')['geometry'].apply(lambda x: LineString(x.tolist()))

        gdf_line.to_file("../shp/cid_shp/car_10000_lines_2.shp", driver='ESRI Shapefile')


    def run(self):

        #self.get_carID()
        cids = pd.read_csv("../csv/car_ID.csv",names=['cid'])['cid'].tolist()
        print(len(cids))
        t0 = time.time()
        cp_ = 0

        # cid = '10035234533245878131'
        # self.get_signle_car(cid)

        self.get_cars(cids)        





if __name__ == '__main__':
    PK = PLOT_KDE()
    PK.run()