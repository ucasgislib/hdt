# -*- coding:utf-8 -*-
"""

"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# warnings.filterwarnings("ignore")
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class TraceProcess(object):


    def __init__(self, data,epsg_target, epsg_origin=4326):
        self.direction = None
        self.speed = None
        self.dist = None
        self.xy = None
        self.gap = None
        self.stay_df = None
        self.move_df = None
        self.region_split = None
        self.trace = data.copy()
        self.trace.lat = data.lat.astype(float)
        self.trace.lng = data.lng.astype(float)
        self.trace.time = pd.to_datetime(data.time)
        self.epsg_origin = epsg_origin
        self.epsg_target = epsg_target
        self.update()

    def update(self):
        """
        更新当前trace表
            gap：时间间隔，
            xy：投影坐标系
            distance：坐标点间距离
            speed：速度
            direction：方向角
        """
        # 时间间隔计算
        gap_s = np.array(np.diff(self.trace.time.astype("datetime64[s]")) * 10 ** -9, dtype=int)  # int数组,单位s
        self.gap = gap_s


        self.xy = gpd.points_from_xy(self.trace.lng, self.trace.lat, crs=self.epsg_origin).to_crs(self.epsg_target)
        self.dist = self.xy[:-1].distance(self.xy[1:])
        self.speed = np.divide(self.dist, gap_s, out=np.zeros_like(
            self.dist), where=gap_s != 0)

        # direction 求取错误
        # self.direction = (np.arctan2(self.xy.y, self.xy.x) * 180 / np.pi).astype(np.int16)

        # 新的计算信息均为后向差分
        self.trace['gap'] = [0, ] + gap_s.tolist()
        self.trace['distance'] = [0, ] + self.dist.tolist()
        self.trace['speed'] = [0, ] + self.speed.tolist()
        # self.trace['direction'] = [0, ] + self.direction.tolist()

        return self.trace

    def stay_move_detect(self, speed_min, dura_second, clips=np.array([])):
        """
        返回的数对为全包含
        数组切片为半包含
        """
        if clips.size:
            speed = self.speed[clips[0]:clips[1]]
            trace = self.trace[clips[0]:clips[1] + 1]
        else:
            speed = self.speed
            trace = self.trace
        # 静止点
        state_arr = speed <= speed_min
        trace_point = np.append(
            state_arr[0], state_arr[1:] ^ state_arr[:-1])  # len=n
        # 取所有状态变化点
        all_point = np.argwhere(trace_point != 0).flatten()

        # all_stay 构建
        if all_point.size:
            # 若存在状态变化
            # all_point：静止路段点索引，为奇数时，则其以静止结尾，需添加末尾轨迹点索引为路程点
            if all_point.size % 2:
                all_point = np.append(all_point, len(state_arr))

            # 筛选静止路程段
            all_stay = all_point.reshape(-1, 2)
            stime = trace.time.iloc[all_stay[:, 0]].values
            etime = trace.time.iloc[all_stay[:, 1]].values
            duration = ((etime - stime) * 10 ** -9).astype(np.int32)
            all_stay = all_stay[duration >= dura_second]

        else:
            # 无状态变化时根据初始状态判定轨迹
            all_stay = np.array([[0, len(state_arr)]]) if state_arr[0] else np.array([[]], dtype=int)

        # all_move 构建
        all_stay_fla = all_stay.flatten()
        if all_stay.size > 0:
            # 静止段以起点开始，则去掉起点
            if all_stay_fla[0] == 0:
                all_stay_fla = all_stay_fla[1:]
            else:
                all_stay_fla = np.concatenate([[0], all_stay_fla])
            # 静止段以终点结尾，则去掉终点
            if all_stay_fla[-1] == len(state_arr):
                all_stay_fla = all_stay_fla[:-1]
            else:
                all_stay_fla = np.concatenate([all_stay_fla, [len(state_arr)]])

            all_move = all_stay_fla.reshape(-1, 2)

        else:
            all_move = np.array([[0, len(trace) - 1]])

        index = np.array([], dtype=int)
        if all_stay.size:
            for i in all_stay:
                index = np.append(index, [x for x in range(i[0], i[1])])

        if clips.size:
            all_move += clips[0]
            all_stay += clips[0]
        return all_stay, all_move, index

    def outlier_detect(self, data=np.array([[]]), neibor_num=300, std_ratio=2.7):
        """
        离群点检测:待验证效果
        @param data:
        @param neibor_num:
        @param std_ratio:
        @return: 索引

        针对二维数据：

            无效点剔除：lng，lat==0
            全局离群点：半径滤波：做不做的吧
            情境离群点：统计方式剔除，距离平均值大于n倍标准差

        """
        out_list = []

        if data.size == 0 and len(self.trace) > 3:
            data = np.array(self.trace[['lng', 'lat']])
        else:
            return []

        for x in range(data.shape[0] // neibor_num):
            c = data[x * neibor_num:(x + 1) * neibor_num]
            st = np.linalg.norm(c.std(axis=0))
            for index, valus in enumerate(c):
                av = np.abs(c - valus).sum(axis=0) / (c.shape[0] - 1)
                av = np.linalg.norm(av)
                if av > std_ratio * st:
                    out_list.append(index + x * neibor_num)
        if data.shape[0] % neibor_num:
            c = data[-neibor_num:]
            st = np.linalg.norm(c.std(axis=0))
            for index, valus in enumerate(c):
                av = np.abs(c - valus).sum(axis=0) / (c.shape[0] - 1)
                av = np.linalg.norm(av)
                if av > std_ratio * st:
                    out_list.append(index + data.shape[0] - neibor_num)
        return np.unique(out_list).tolist()

    def drift_detect(self, speed_max=None):
        """
            超速点检测
        """
        if not speed_max:
            speed_max = np.percentile(self.speed, 99.8)

        index_out_limit = np.argwhere(self.speed > speed_max).reshape(-1, )
        if (index_out_limit.size) == 0:
            return 0

        # print('ind', index_out_limit)
        # 记录该删除的索引
        del_list = (index_out_limit + 1).tolist()

        # 遍历速度异常值的索引
        for index in index_out_limit.tolist():
            id_tmp = self.trace.cid[index]

            # 若因轨迹结束，或车辆id改变造成速度异常，则不进行操作
            if index == len(self.trace) - 2 or id_tmp != self.trace.cid[index + 1]:
                del_list.remove(index + 1)
                # print("change")
                continue

            # 速度不满足条件时record依次后推
            record = index + 2

            while True:
                # 轨迹超出，或下一点车辆id变更时，不进行操作
                if record == len(self.trace) or id_tmp != self.trace.cid[record]:
                    break
                # 求俩点间速度
                gap_tmp = np.int32(
                    self.trace.time.values[record] - self.trace.time.values[index]) * 10 ** -9
                dist_tmp = np.linalg.norm(self.xy[record] - self.xy[index])
                speed_tmp = dist_tmp / gap_tmp if gap_tmp != 0 else 0
                #                 spe=np.linalg.norm(self.xy[record]-self.xy[index])/np.int32((self.trace.time.values[record]-self.trace.time.values[index])*10**-9)
                if speed_tmp < speed_max:
                    break
                del_list.append(record)
                record += 1

        # print("del", del_list)
        self.del_list = del_list
        return del_list

    def clean_point(self, index_points):
        """
        清除索引点，更新参数
        """
        # 丢弃异常值，重设索引
        self.trace.drop(index=index_points, inplace=True)
        self.trace.reset_index(drop=True, inplace=True)
        self.update()
        return len(self.trace)

    def visual_data(self):
        '''
        可视化
        '''
        trance = self.trace
        # 有关数据间隔，速度，方向角的有效数据表(删除表中不同辆车间产生的数据)
        info_valid = trance[1:][trance.cid[1:].values == trance.cid[:-1].values]
        gap_counts = info_valid.gap.value_counts()  # 时间间隔数据统计
        print("时间间隔频次排名前十", gap_counts.head(10))
        # plt.figure(figsize=(10,5),dpi=300)
        # # plt.tight_layout()  #子图紧凑
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=2, hspace=0.2)#子图间隔
        # fig,(ax1,ax2)=plt.subplots(2,1)
        gap_upper_limit = np.percentile(
            np.array(info_valid.gap), 90)  # 数据直方图中数据上限取90百分位
        plt.figure(figsize=(10, 4), dpi=300)
        plt.hist(info_valid.gap,
                 bins=None,
                 range=(0, gap_upper_limit),
                 density=False,
                 weights=None,
                 cumulative=False,
                 bottom=None,
                 histtype='bar',
                 align='mid',
                 orientation='vertical',
                 rwidth=0.3,
                 log=False,
                 color=None,
                 label=None,
                 stacked=False,
                 )
        plt.xlabel("时间间隔：s")
        plt.ylabel("次")
        plt.title("时间间隔直方图")

        # ditance_view = info_valid.distance[~(np.isnan(info_valid.distance) | np.isinf(info_valid.distance))]
        # distance_upper_limit = np.percentile(ditance_view, 95)  # 数据直方图中数据上限取90百分位
        # #     speed_counts=info_valid.distance.value_counts()#速度数据统计
        #
        # # 选取距离小于 阈值 的数据进行展示
        # plt.figure(figsize=(10, 4), dpi=300, )
        # plt.hist(ditance_view, bins=21, range=(0, 300), rwidth=0.8)
        # plt.ylabel("间距：m")
        # plt.xticks([x * 10 for x in range(30)])
        # plt.title("间距折线图")
        # print("distance NAN %s 条" % (info_valid.distance.size - ditance_view.size))


def bp_detect(old_index):
    """
    :param trace: 大于3的gps表
    :param old_index_name: 旧索引列名
    :return continue_point:
    """
    index = old_index
    state = index[1:] - index[:-1]
    state[np.where(state != 1)] = 0

    # 对状态列表 和 下一状态表 做异或，得到状态变化点trace_point
    trace_point = np.append(
        state[0], state[1:] ^ state[:-1])  # len=n
    # 取所有状态变化点
    all_point = np.argwhere(trace_point == 1).flatten()
    # all_stay 构建
    if all_point.size:
        # 若存在状态变化
        # all_point：连续路段点索引，为奇数时，则其以连续状态结尾，需添加末尾轨迹点索引为路程点
        if all_point.size % 2:
            all_point = np.append(all_point, len(state))

        # 筛选静止路程段
        continue_point = all_point.reshape(-1, 2)
    else:
        # 无状态变化时根据初始状态判定轨迹
        continue_point = np.array([[0, len(state)]]) if state[0] else np.array([[]])
    return continue_point


def keep_region(data, region_map, espg_trace=4326):
    """

    需要构建point
    @param data:
    @param region_map:
    @param espg_trace:
    @return:
    """

    map_shape = region_map
    trace = data.copy()
    trace_gdf = gpd.GeoDataFrame(trace, geometry=gpd.points_from_xy(trace.lng, trace.lat), crs=espg_trace)
    sjoin_data = gpd.sjoin(map_shape, trace_gdf, predicate="contains")
    sjoin_data.sort_values('index_right', inplace=True)
    keep_index = sjoin_data.index_right.values
    if keep_index.size > 3:
        return keep_index, bp_detect(keep_index)
    return keep_index, np.array([[]])


def clip_traj(processer, trace_keep, points_bp, gap_max, dis_max):
    """

    @param processer:
    @param trace_keep:
    @param points_bp:
    @param gap_max:
    @param dis_max:
    @return:
    """
    bp_arr = np.array([], dtype=int)
    for i in points_bp:
        pp = trace_keep[i]
        cut_s = trace_keep[i][0] + np.where((processer.gap[trace_keep[i][0]:trace_keep[i][1]] > gap_max) | (
                processer.dist[trace_keep[i][0]:trace_keep[i][1]] > dis_max))
        cut_s = np.unique(cut_s)
        cut_e = cut_s + 1
        pp = np.append(pp, (cut_s, cut_e))
        # print(pp)
        bp_arr = np.append(bp_arr, pp)
    bp_arr.sort()
    bp_arr = bp_arr.reshape(-1, 2)
    bp_arr = bp_arr[bp_arr[:, 1] - bp_arr[:, 0] > 3]
    return bp_arr

