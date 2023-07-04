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
from sqlalchemy import create_engine
from sqlalchemy import select,Table
from sqlalchemy import func,extract
from sqlalchemy.orm import aliased,registry

import datetime
import pytz
from PIL import ImageTk
ImageTk.Tkinter.TkImage = ImageTk.PhotoImage

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

    def get_carID(self):

        # 执行 SQL 查询
        self.cur.execute("SELECT distinct(cid) FROM %s "%self.t_name)

        # 获取所有结果行
        cars = self.cur.fetchall()

        # 输出每一行的结果
        fw = open("../csv/car_ID.csv","w")
        for row in cars:
            fw.write(row[0]+"\n")
        fw.close()
    def init_proj(self):
        params = {'proj': 'merc', 'a': 6378137, 'b': 6378137, 'lat_ts': 0, 'lon_0': 0, 'x_0': 0, 'y_0': 0, 'k': 1.0, 'units': 'm', 'nadgrids': '@null'}
        self.proj3857 = Proj(params)

    def gaussian(self, h, dists, dist_):
        A = ((2*np.pi)**(-1/2))
        u = (dists-dist_)/h
        k = A*np.exp((-1/2)*(u**2))
        return k

    def cal_kde(self, trj, h=0.5, step=0.05):
        trj = trj.sort_values(by='ts').values
        dists_ = np.arange(0,self.max_dist,step)
        tsum = ((trj[-1][1]-trj[0][1])/1000)
        stays, dists = [],[]
        for cid,gp in groupby(trj, key=lambda x:x[2]):

            gp = list(gp)
            spd_,tim_,cid_,lon0,lat0 = gp[0]
            x_,y_ = self.proj3857(lon0,lat0)
            for i,rec in enumerate(gp):
                if i == 0 or i == len(gp):
                    continue
                
                spd,tim,cid,lon,lat = rec
                staytime = int((tim - tim_)/1000)
                x,y = self.proj3857(lon,lat)
                dist = np.sqrt((x-x_)**2+(y-y_)**2)/1000
                stays.append(staytime)
                dists.append(dist)
                
                spd_,tim_,cid_,lon_,lat_ = rec
        stays = np.array(stays)
        w = (1+np.array(stays))/sum((1+np.array(stays)))

        scores = []
        for dist_ in dists_:
            k = self.gaussian(h,dists,dist_)
            score = np.mean(w*k)/h
            scores.append(score)

        return dists_,scores

    def plot_dist(self, trj):
        ts,lons,lats = trj['ts'],trj['lon'],trj['lat']
        self.x0,self.y0 = self.proj3857(lons[0],lats[0])
        func = lambda p:self.proj3857(p[0],p[1])
        xys = np.array(list(map(func,np.c_[lons,lats])))
        xs,ys = xys[:,0],xys[:,1]
        ts = ts/1000
        ds = np.sqrt((xys[:,0]-self.x0)**2+(xys[:,1]-self.y0)**2)/1000.0

        fig,ax = plt.subplots(figsize=(12, 8))
        ax.plot(ts, ds, lw=1, c="blue", ls='-', zorder=1)
        ax.scatter(ts, ds, s=30, c="blue", zorder=2)

        # t0 = int(time.mktime(time.strptime("2016-08-04 00:00:00", "%Y-%m-%d %H:%M:%S")))
        # t1 = int(time.mktime(time.strptime("2016-08-05 00:00:01", "%Y-%m-%d %H:%M:%S")))
        # tticks = np.arange(t0,t1,3600*4)
        # tticklables = list(map(lambda x:time.strftime("%H:%M",time.localtime(x)), tticks))
        # tticklables[-1] = "24:00"
        # ax.set_xticks(tticks)
        # ax.set_xticklabels(tticklables, rotation=30)
        # ax.set_ylim(0,self.max_dist)
        # ax.grid(zorder=0)

        # plt.savefig("./dist_blue.png")
        plt.show()


    def plot_kde(self, trj):
        # print(trj)
        dists_, kde_scores = self.cal_kde(trj)

        fig,ax = plt.subplots(figsize=(12, 8))
        ax.plot(dists_, kde_scores, lw=3, c="black", ls='-', zorder=1)
        ax.grid(zorder=0)
        plt.show()

    def plot_all(self, cid, df):

        import pandas as pd
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        df['date'] = df['tim'].dt.date

        # 将每个GPS点转换为立方体，并设置高度为停留时间长度
        cubes = []
        for row in df.itertuples():
            cube = [
                [row.lon, row.lat, row.date],
                # [row.lon, row.lat, row.date ],
                # [row.lon , row.lat, row.date],  # + pd.DateOffset(days=1)
                # [row.lon, row.lat, row.date],
                # [row.lon, row.lat , row.date],
                # [row.lon, row.lat, row.date ],
                # [row.lon , row.lat, row.date],
                # [row.lon + 0.0005, row.lat + 0.0005, row.date]
            ]
            cubes.append(cube)

        # 根据日期对立方体分组并着色
        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        colors = ['#8F8F8F', '#6B8E23', '#BC8F8F', '#ADD8E6', '#BA55D3', '#CD853F', '#FFDAB9', '#FFE4C4', '#FFFFF0', '#A9A9A9']

        groups = df.groupby('date')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, (name, group) in enumerate(groups):
            xs = []
            ys = []
            zs = []
            for cube in cubes:
                if cube[0][2] == name:
                    for vertex in cube:
                        xs.append(vertex[0])
                        ys.append(vertex[1])
                        zs.append(vertex[2].toordinal())
            ax.plot(xs, ys, zs, c=colors[i % len(colors)])
            # ax.scatter(xs, ys, zs, c=colors[i % len(colors)], marker='s',s=7)

        # 设置坐标轴标签
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Time')

        # 显示图形
        # plt.show()
        plt.savefig("../png/ax3d/ax3d_%s.png"%cid)
        plt.close('all')

    def run(self):
        #self.get_carID()
        cids = pd.read_csv("../csv/car_ID.csv",names=['cid'])['cid'].tolist()
        print(len(cids))
        cp = 0
        xmin, xmax = 105.18, 111.15   # 经度范围
        ymin, ymax = 31.42, 39.35     # 纬度范围
        for cid in cids:
            sql = "select * from %s where cid = '%s' "%(self.t_name,cid)
            self.cur.execute(sql)

            # df = pd.read_sql(sql,self.engine)
            # print(df.head())
            traj = self.cur.fetchall()
            # print("-------------------------------------------")
            # print(cid,len(traj))
            if len(traj) > 7:
                try:
                    trj = pd.DataFrame(traj, columns=['cid', 'tim', 'lon', 'lat', 'spe', 'speed', 'legend', 'height', 'direction', 'status', 'warn'])

                    mask = (trj['lon'] >= xmin) & (trj['lon'] <= xmax) & (trj['lat'] >= ymin) & (trj['lat'] <= ymax)
                    trj = trj.loc[mask]


                    trj['ts'] = trj['tim'].apply(lambda x: pd.Timestamp(x).timestamp())

                    sel_cols = ['speed','ts','cid','lon','lat','tim']
                    trj = trj[sel_cols]
                    trj = trj.sort_values(by='ts')
                    # print(trj.values[0])
                    self.plot_all(cid,trj)

                    # exit()
                    # self.plot_kde(trj)
                    # self.plot_dist(trj)

                    cp+=1
                    if cp %100 == 0:
                        print(cp)
                        # exit()
                except:
                    pass


if __name__ == '__main__':
    PK = PLOT_KDE()
    PK.run()