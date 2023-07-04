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

from pyproj import Proj
from geopy.distance import great_circle
from shapely.geometry  import MultiPoint,Polygon
from geopy.geocoders  import Nominatim



plt.rcParams['font.family'] = 'Times New Roman' # 设置字体样式
plt.rcParams['font.sans-serif'] = 'SimHei'

plt.rcParams['figure.figsize'] = (6,6)     # 显示图像的最大范围
# plt.rcParams['figure.figsize'] = (8, 6)     # 显示图像的最大范围

plt.rcParams['font.size'] = 16 # 设置字体大小
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['savefig.dpi'] = 600

class PLOT_KDE(object):
    """docstring for GPS_KDE"""
    def __init__(self):
        pass

    def __del__(self):
        pass

    def get_region(self):
        xian_gpd = gpd.read_file("../shp/边界/陕西省县级市.shp")
        xian_dict = xian_gpd.set_index('name')['adcode'].to_dict()
        print(xian_dict)

        shi_gpd = gpd.read_file("../shp/边界/陕西省地级市.shp")
        shi_dict = shi_gpd.set_index('name')['adcode'].to_dict()
        print(shi_dict)

        return shi_dict,xian_dict


    def stat_xian(self):

        od_county_ocount = pd.read_csv("../wang/chart/od_county_ocount.csv")

        od_county_dcount = pd.read_csv("../wang/chart/od_county_dcount.csv")

        od_county_ocount['ods'] = od_county_ocount['ct']+od_county_dcount['ct']

        print(od_county_ocount.columns)

        gdp = pd.read_excel("../统计年鉴/各县生产总值（2021）.xls", usecols="A:P", nrows=29, skiprows=8,names=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"])
        gdp_dict = {}
        # 遍历DataFrame的每一行
        for index, row in gdp.iterrows():
            if not pd.isna(row['C']):  
                name = row['A'].replace(" ", "")
                gdp_dict[name] = row['C']

            if not pd.isna(row['G']):  
                name = row['E'].replace(" ", "")
                gdp_dict[name] = row['G']

            if not pd.isna(row['K']):  
                name = row['I'].replace(" ", "")
                gdp_dict[name] = row['K']

            if not pd.isna(row['O']):  
                name = row['M'].replace(" ", "")
                gdp_dict[name] = row['O']

        pop = pd.read_excel("../统计年鉴/各县常住人口（2021）.xls", usecols="A:P", nrows=29, skiprows=5,names=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"])
        print(pop)
        pop_dict = {}
        # 遍历DataFrame的每一行
        for index, row in pop.iterrows():
            if not pd.isna(row['D']):  
                name = row['A'].replace(" ", "")
                pop_dict[name] = row['D']

            if not pd.isna(row['H']):  
                name = row['E'].replace(" ", "")
                pop_dict[name] = row['H']

            if not pd.isna(row['L']):  
                name = row['I'].replace(" ", "")
                pop_dict[name] = row['L']

            if not pd.isna(row['P']):  
                name = row['M'].replace(" ", "")
                pop_dict[name] = row['P']

        comp = pd.read_excel("../统计年鉴/各县单位法人数（2021）.xls", usecols="A:L", skiprows=8,names=["A","B","C","D","E","F","G","H","I","J","K","L"])
        print(comp)
        comp_dict = {}
        # 遍历DataFrame的每一行
        for index, row in comp.iterrows():
            if not pd.isna(row['C']):  
                name = row['A'].replace(" ", "")
                comp_dict[name] = row['C']

            if not pd.isna(row['F']):  
                name = row['D'].replace(" ", "")
                comp_dict[name] = row['F']

            if not pd.isna(row['I']):  
                name = row['G'].replace(" ", "")
                comp_dict[name] = row['I']

            if not pd.isna(row['L']):  
                name = row['J'].replace(" ", "")
                comp_dict[name] = row['L']

        od_county_ocount.set_index('name',inplace= True)
        

        gdp_df = pd.DataFrame.from_dict(gdp_dict, orient='index')
        gdp_df.columns = ['gdp']

        pop_df = pd.DataFrame.from_dict(pop_dict, orient='index')
        pop_df.columns = ['pop']

        comp_df = pd.DataFrame.from_dict(comp_dict, orient='index')
        comp_df.columns = ['comp']
        
        result_df = od_county_ocount.merge(gdp_df, left_index=True, right_index=True)
        result_df = result_df.merge(pop_df, left_index=True, right_index=True)
        result_df = result_df.merge(comp_df, left_index=True, right_index=True)

        # 计算相关系数
        corr_gdp = result_df['ods'].corr(result_df['gdp'])
        corr_pop = result_df['ods'].corr(result_df['pop'])
        corr_comp = result_df['ods'].corr(result_df['comp'])

        plt.scatter(result_df['ods'], result_df['gdp'])
        plt.xlabel('OD Number')
        plt.ylabel('GDP')
        plt.title("Correlation Coefficient: %s"%corr_gdp)
        plt.show()

        plt.scatter(result_df['ods'], result_df['pop'])
        plt.xlabel('OD Number')
        plt.ylabel('POP')
        plt.title("Correlation Coefficient: %s"%corr_pop)
        plt.show()

        plt.scatter(result_df['ods'], result_df['comp'])
        plt.xlabel('OD Number')
        plt.ylabel('Company Number')
        plt.title("Correlation Coefficient: %s"%corr_comp)
        plt.show()

        result_df.to_csv("../csv/corr.csv")


    def run(self):


        shi_dict,xian_dict = self.get_region()

        self.stat_xian()

        # od_city_ocount = pd.read_csv("../wang/chart/od_city_ocount.csv")

        # od_city_dcount = pd.read_csv("../wang/chart/od_city_dcount.csv")

        # od_city_ocount['ods'] = od_city_ocount['ct']+od_city_dcount['ct']

        # print(od_city_ocount)

        

if __name__ == '__main__':
    PK = PLOT_KDE()
    PK.run()