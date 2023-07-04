import os
import pandas as pd
import re
import geopandas as gpd
from shapely.geometry import Point

pp=pd.DataFrame()
inpath="../2022/陕西省POI点-csv/GBK/"


keys = ["货运","卡车","重卡","运输","物流","化工",
        "国际港","维修","冶炼","铁矿","铝矿","锡矿","钢厂","煤炭","选煤","商砼",
        "陕汽","吉利","煤矿","发电厂","物流园","混凝土","比亚迪","厂区","产业园",
        "物流中心","采购中心","公路港","分拣中心","集散","仓库","运营中心","有色","水泥",
        "煤业","洗煤","检验检测","能源化工"]

key = "|".join(keys)


for val in os.listdir(inpath):
    # if re.search("公司企业|汽车",val):
    print(val)
    csv_path=os.path.join(inpath,val)
    tmp=pd.read_csv(csv_path,encoding='gb18030',low_memory=False)
    poi_place=tmp.iloc[tmp.apply(lambda col:re.search(key,col["name"]),axis=1).dropna().index]
    pp=pd.concat([pp,poi_place])
pp.name
print(pp.columns)


geometry = [Point(xy) for xy in zip(pp.longitude_wgs84, pp.latitude_wgs84)]


gdf = gpd.GeoDataFrame(pp['name'], geometry=geometry)

gdf.to_file('../shp/货运相关POI.shp', driver='ESRI Shapefile')