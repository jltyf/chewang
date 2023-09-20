import bisect
import math
import os
import time
import datetime
import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from scenariogeneration import xosc
from enum import Enum

import xml.etree.ElementTree as ET

# import matplotlib.pyplot as plt
from scipy import signal

# offset_x = -459139.28117948445
# offset_y = -4406370.951575764

crs = CRS.from_epsg(4326)
crs_cs = CRS.from_epsg(4509)
# crs_cs = CRS.from_epsg(32650)

transformer = Transformer.from_crs(crs, crs_cs)


class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                 (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
                 self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B


def get_coordinate(s):
    longitude = s['longitude']
    latitude = s['latitude']
    y, x = transformer.transform(latitude, longitude)
    return x, y


class WorkMode(Enum):
    roadside = 1
    car = 2
    merge = 3


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class BIsPoint(object):
    def __init__(self, b, x=0, y=0):
        self.x = x
        self.y = y
        self.b = b


class ObsPosition(object):
    def __init__(self, time_stamp=0, ObjectID=0, ObjectType=0, x=0.0, y=0.0, z=0.0, h=0.0, vel=0.0):
        self.time = time_stamp
        self.ObjectID = ObjectID
        self.ObjectType = ObjectType
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.h = h
        self.vel = vel


x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


def wgs84_to_gcj02(s):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    lng = s['longitude']
    lat = s['latitude']
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    y, x = transformer.transform(mglat, mglng)
    return x, y


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


def get_file(path, key_file_name):
    files = os.listdir(path)

    file_list = []
    if key_file_name not in files:
        for name in files:
            next_path = os.path.join(path, name)
            if os.path.isdir(next_path):
                file_list.extend(get_file(next_path, key_file_name))
    else:
        file_list.append(path)
    file_list = list(set(file_list))

    return file_list


def change_heading(data_df, mode):
    diff = data_df['heading'] - data_df['heading'].shift(1)
    data_df['diff'] = diff.abs()
    data_df['diff'] = data_df['diff'].apply(lambda x: x - 360 if x >= 180 else x)
    diff_data = data_df[abs(data_df['diff']) >= 25]
    # if len(diff_data) == 0:
    #     return data_df
    if mode == 1:
        error_data = data_df[(data_df['diff'] > 25) & (data_df['distance'] < 0.7)]
        if len(error_data) > 1:
            error_start = data_df[data_df['time'] == error_data['time'].min()].index[0]
            error_end = data_df[data_df['time'] == error_data['time'].max()].index[0]
            error_df = data_df[error_start:error_end]
            # if
            right_heading = data_df.loc[error_start - 1, 'heading']
            error_df['heading'] = right_heading
            data_df[error_start:error_end] = error_df
            data_df = change_heading(data_df, mode=1)
        elif len(error_data) == 1:
            error_start = data_df[data_df['time'] == error_data['time'].min()].index[0]
            left_df = data_df[error_start:]
            if left_df[1:]['diff'].mean() < 15:
                left_df['heading'] = data_df.loc[error_start - 1, 'heading']
                data_df[error_start:] = left_df
                data_df = change_heading(data_df, mode=1)
    elif mode == 0:
        error_start = data_df[data_df['time'] == diff_data['time'].min()].index[0] - 2
        error_end = data_df[data_df['time'] == diff_data['time'].max()].index[0] + 2
        error_df = data_df[error_start:error_end]
        if data_df['heading'].var() > 4000:
            return data_df
        if len(error_df[(error_df['distance']) >= 0.7]) < 3:
            filter_data = data_df[(data_df['diff']) <= 1]
            diff = filter_data['heading'] - filter_data['heading'].shift(1)
            filter_data['diff'] = diff.abs()
            avg_heading = filter_data[(filter_data['diff']) <= 1]['heading'].mean()
            error_df['heading'] = avg_heading
            data_df[error_start:error_end] = error_df
        else:
            _ = error_df[1:-1]
            _['heading'] = np.nan
            error_df[1:-1] = _
            error_df = error_df.interpolate()
            data_df[error_start:error_end] = error_df
    return data_df


def filter_error(data_df):
    data_df['distance'] = ((data_df['x'] - data_df['x'].shift(1)) ** 2 + (
            data_df['y'] - data_df['y'].shift(1)) ** 2) ** 0.5
    data_df = data_df.fillna(0)
    diff = data_df['heading'] - data_df['heading'].shift(1)
    data_df['diff'] = diff.abs()
    diff_data = data_df[abs(data_df['diff']) >= 25]
    if len(diff_data) >= 1:
        if (datetime.timedelta(milliseconds=300) < abs(
                diff_data['time'].min() - diff_data['time'].max()) < datetime.timedelta(milliseconds=8000)):
            data_df = change_heading(data_df, mode=0)
        else:
            data_df = change_heading(data_df, mode=1)

    # if data_df.loc[1, 'time'] - data_df.loc[0, 'time'] > datetime.timedelta(seconds=4):
    #     data_df = data_df[1:]

    return data_df


def convert(s):
    x = s.to_pydatetime()

    timeStamp = int(time.mktime(x.timetuple()) * 1000.0 + x.microsecond / 1000.0)

    return timeStamp


def read_obs(obsList, time_list):
    position = []

    for result in obsList:

        if len(result) > 0:
            position.append(
                ObsPosition((float(result[0]) - time_list[0]) / 1000, str(result[1]), result[2], result[3],
                            float(result[4]), 0, math.radians(float(result[5]) + 90)))
    return position


def read_gps(path):
    data = pd.read_csv(path)
    position = []
    time_list = []
    # data = data[['Time', 'East', 'North', 'HeadingAngle']]
    # data = data.loc[data['ID'] == -1, ['Time', 'East', 'North', 'HeadingAngle']].ffill(inplace=True)
    data = data.loc[data['ID'] == -1, ['Time', 'East', 'North', 'HeadingAngle']]
    data = data.reset_index(drop=True)
    data.ffill(inplace=True)

    # x = float(data.at[0, 'East'])
    # y = float(data.at[0, 'North'])
    x = 0
    y = 0
    for row in data.iterrows():
        position.append(xosc.WorldPosition(x=float(row[1]['East']) - x, y=float(row[1]['North']) - y,
                                           h=math.radians(float(row[1]['HeadingAngle']) + 90)))
        time_list.append(float(row[1]['Time']))
    return position, x, y, time_list


def read_gps_c(obsList, time_list):
    position = []

    for result in obsList:
        z = 0
        new_position = ObsPosition(time_stamp=((result[0] - time_list[0]) / 1000), ObjectID=result[2],
                                   ObjectType=result[1], x=float(result[3]), y=float(result[4]), z=float(z),
                                   h=float(math.radians(result[5])))
        position.append(new_position)
    return position


def read_gps_lu(obsList, time_list):
    position = []
    for result in obsList:
        time_now = (result[0] - time_list[0]) / 1000
        h = math.radians(float(90 - result[6]))
        z = result[5] / 10
        position.append(
            ObsPosition(time_now, str(result[1]), result[2], float(result[3]), float(result[4]), z, h))
    return position


def load_data(csvPath, obsPath):
    data = pd.read_csv(csvPath)
    data = data[data['Type'] != 10]  # 排除非目标物的物体
    data = data.loc[data['ID'] != -1, ['Time', 'ID', 'East', 'North', 'HeadingAngle', 'AbsVel', 'Type']]

    obs_data = pd.read_csv(obsPath)
    obs_data = obs_data[
        (obs_data['ObjectPosY'] < 5) & (obs_data['ObjectPosY'] > -5) & (obs_data['ObjectPosX'] > -30) & (
                obs_data['ObjectPosX'] < 100)]  # 排除车道线范围外且前向距离较远的目标物
    id_list = obs_data['ObjectID'].tolist()  # 筛选出符合条件的的目标物ID
    data = data[(data['ID'].isin(id_list))]
    for obj_id, obj_data in data.groupby('ID'):
        rating = obj_data['HeadingAngle'] - obj_data['HeadingAngle'].shift(3)
        obj_data['rating'] = rating.abs()
        obj_data.mask(cond=obj_data['rating'] > 1, other=obj_data.shift(3), inplace=True)
        data.loc[data['ID'] == obj_id] = obj_data

    data = data.reset_index(drop=True)

    obs_data = data.copy()
    obs_data = obs_data[['Time', 'ID', 'Type', 'East', 'North', 'HeadingAngle']]
    groups = obs_data.groupby('ID')
    obs_list = []
    for key, value in groups:
        obs_list.append(value.values.tolist())

    return obs_list


# def load_data_lu(ego_path, obs_path, target_number_list, target_area, offset_list):
#     """
#     Load data from single folder(work mode is roadside)
#     :param ego_path: the path of file that record ego's position
#     :param obs_path: the path of file that record objects' position
#     :param target_number_list: set of ego plate number
#     :param target_area: area number
#     :param offset_list: record different area' offset of OpenDrive
#     :return:result tuple, include:(ego_position:list, obs_list:list, time_list:list, init_speed:int)
#     """
#     ego_id_list = list()
#
#     # 新的csv格式数据
#     pos_data = pd.read_csv(obs_path)
#     pos_data.rename(
#         columns={'时间戳': 'datetime', '感知目标ID': 'id', '感知目标经度': 'longitude', '感知目标纬度': 'latitude',
#                  '高程(dm)': 'altitude', '速度(m/s)': 'speed', '航向角(deg)': 'heading', '感知目标类型': 'type'},
#         inplace=True)
#     pos_data['time'] = pd.to_datetime(pos_data['datetime'])
#     pos_data = pos_data[['time', 'id', 'type', 'longitude', 'latitude', 'speed', 'heading', 'altitude']]
#     pos_data['id'] = pos_data['id'].astype('str')
#     pos_data.id = pos_data.id.apply(lambda x: x[-10:])
#
#     for target_number in target_number_list:
#         if target_number in pos_data['id'].values:
#             ego_id_list = target_number_list
#             break
#
#     # offset_x, offset_y = (-1, -1)
#     # for offset in offset_list:
#     #     if target_area == offset[4:7]:
#     #         offset_value = offset.split(':')[1]
#     #         offset_x = float(offset_value.split(',')[0])
#     #         offset_y = float(offset_value.split(',')[1])
#     #         break
#     if not ego_id_list:
#         return 401
#     elif offset_x == -1 and offset_y == -1:
#         return 402
#     pos_data = pos_data.reset_index(drop=True)
#     obs_data = pos_data
#     for ego_id in ego_id_list:
#         obs_data = obs_data[obs_data['id'] != ego_id].reset_index(drop=True)
#
#     # 使用车端上传的数据
#     ego_data = pd.read_csv(ego_path)
#     ego_data.rename(
#         columns={'时间戳': 'datetime', '经度': 'longitude', '纬度': 'latitude', '高程(dm)': 'altitude', '速度(m/s)': 'speed',
#                  '航向角(deg)': 'heading'}, inplace=True)
#     ego_data['time'] = pd.to_datetime(ego_data['datetime'])
#
#     ego_data['heading'] = ego_data['heading'].astype('float')
#     start_time = ego_data['time'].min()
#     end_time = ego_data['time'].max()
#     obs_data = obs_data[(obs_data['time'] >= start_time) & (obs_data['time'] <= end_time)].reset_index(
#         drop=True)  # 记录结束时间
#     # ego_data[['x', 'y']] = ego_data.apply(get_coordinate, axis=1, result_type='expand')
#     ego_data[['x', 'y']] = ego_data.apply(wgs84_to_gcj02, axis=1, result_type='expand')
#     init_speed = np.mean(ego_data.loc[0:4, 'speed'].values.tolist())
#     ego_data = ego_data[['time', 'x', 'y', 'heading', 'altitude']]
#     ego_data['x'] = ego_data['x'] + offset_x
#     ego_data['y'] = ego_data['y'] + offset_y
#
#     # plt_trail(ego_data['x'].values.tolist(), ego_data['y'].values.tolist())
#
#     ego_data['date_time'] = pd.to_datetime(ego_data['time'], unit='ms')
#     ego_data = ego_data.resample('100ms', on='date_time').mean().bfill()
#     ego_data = filter_error(ego_data)
#
#     ego_data['tmp'] = ego_data.index
#     time_list = (ego_data.tmp.apply(lambda x: convert(x))).values.tolist()
#
#     # plt_trail(ego_data['x'].values.tolist(), ego_data['y'].values.tolist())
#
#     obs_data['date_time'] = pd.to_datetime(obs_data['time'], unit='ms')
#     obs_data[['x', 'y']] = obs_data.apply(wgs84_to_gcj02, axis=1, result_type='expand')
#     # obs_data[['x', 'y']] = obs_data.apply(get_coordinate, axis=1, result_type='expand')
#     obs_data['x'] = obs_data['x'] + offset_x
#     obs_data['y'] = obs_data['y'] + offset_y
#     obs_data['id'] = obs_data['id'].astype('int64')
#     obs_data['type'] = obs_data['type'].astype('float')
#     obs_data['heading'] = obs_data['heading'].astype('float')
#     groups = obs_data.groupby('id')
#     obs_list = []
#
#     for obj_id, obs_df in groups:
#         if len(obs_df) < 7 or (
#                 obs_df.iloc[-1]['time'] - obs_df.iloc[-0]['time'] < datetime.timedelta(milliseconds=700)):
#             continue
#
#         obs_df = obs_df.reset_index(drop=True)
#
#         # # for test
#         # if obj_id == 3838623030:
#         #     print(110)
#         #     print(110)
#
#         obs_df = filter_error(obs_df)
#         obs_df = obs_df.resample('100ms', on='date_time').mean()
#         obs_df.dropna(inplace=True, axis=0)
#         obs_df['time'] = obs_df.index
#         obs_df['time'] = obs_df.time.apply(lambda x: convert(x))
#         obs_df = obs_df[['time', 'id', 'type', 'x', 'y', 'altitude', 'heading']]
#         if len(obs_df) > 27:
#             sos = signal.butter(8, 0.2, 'lowpass', output='sos')
#             filtered_x = signal.sosfiltfilt(sos, obs_df['x'])
#             filtered_y = signal.sosfiltfilt(sos, obs_df['y'])
#             filtered_h = signal.sosfiltfilt(sos, obs_df['heading'])
#             obs_df['x'] = signal.savgol_filter(filtered_x, 15, 2, mode='nearest')
#             obs_df['y'] = signal.savgol_filter(filtered_y, 15, 2, mode='nearest')
#             obs_df['heading'] = signal.savgol_filter(filtered_h, 15, 2, mode='nearest')
#         if len(obs_df) < 7:
#             continue
#         result_df = pd.merge(obs_df, ego_data, left_index=True, right_index=True, how='left')
#         result_df['distance'] = result_df.apply(euclidean_distance, axis=1, result_type='expand')
#         distance_count = result_df.loc[result_df['distance'] <= 1.2]
#
#         if len(distance_count) > 3:
#             continue
#         obs_list.append(obs_df.values.tolist())
#     ego_data['tmp'] = pd.to_datetime(ego_data['tmp'])
#     ego_data['tmp'] = ego_data['tmp'].astype('int64')
#     gps = ego_data.values.tolist()
#     ego_position = list()
#     for result in gps:
#         if len(result) > 0:
#             ego_position.append(
#                 xosc.WorldPosition(x=float(result[1]), y=float(result[2]),
#                                    z=float(result[4]) / 10, h=math.radians(float(90 - float(result[3])))))
#     return ego_position, obs_list, time_list, init_speed

def load_data_lu(ego_path, obs_path, target_number_list, target_area, offset_list):
    """
    Load data from single folder(work mode is roadside)
    :param ego_path: the path of file that record ego's position
    :param obs_path: the path of file that record objects' position
    :param target_number_list: set of ego plate number
    :param target_area: area number
    :param offset_list: record different area' offset of OpenDrive
    :return:result tuple, include:(ego_position:list, obs_list:list, time_list:list, init_speed:int)
    """
    ego_id_list = list()

    # 新的csv格式数据
    pos_data = pd.read_csv(obs_path)
    pos_data.rename(
        columns={'时间戳': 'datetime', '感知目标ID': 'id', '感知目标经度': 'longitude', '感知目标纬度': 'latitude',
                 '高程(dm)': 'altitude', '速度(m/s)': 'speed', '航向角(deg)': 'heading', '感知目标类型': 'type'},
        inplace=True)
    pos_data['time'] = pd.to_datetime(pos_data['datetime'])
    pos_data = pos_data[['time', 'id', 'type', 'longitude', 'latitude', 'speed', 'heading', 'altitude']]
    pos_data['id'] = pos_data['id'].astype('str')
    pos_data.id = pos_data.id.apply(lambda x: x[-10:])

    for target_number in target_number_list:
        if target_number in pos_data['id'].values:
            ego_id_list = target_number_list
            break

    offset_x, offset_y = (-1, -1)
    for offset in offset_list:
        if target_area == offset[4:7]:
            offset_value = offset.split(':')[1]
            offset_x = float(offset_value.split(',')[0])
            offset_y = float(offset_value.split(',')[1])
            break
    if not ego_id_list:
        return 401
    elif offset_x == -1 and offset_y == -1:
        return 402
    pos_data = pos_data.reset_index(drop=True)
    obs_data = pos_data
    for ego_id in ego_id_list:
        obs_data = obs_data[obs_data['id'] != ego_id].reset_index(drop=True)

    # 使用车端上传的数据
    ego_data = pd.read_csv(ego_path)
    ego_data.rename(
        columns={'时间戳': 'datetime', '感知目标ID': 'id', '经度': 'longitude', '纬度': 'latitude', '高程(dm)': 'altitude',
                 '速度(m/s)': 'speed', '航向角(deg)': 'heading', '感知目标类型': 'type'}, inplace=True)
    ego_data['time'] = pd.to_datetime(ego_data['datetime'])

    ego_data['heading'] = ego_data['heading'].astype('float')
    start_time = ego_data['time'].min()
    end_time = ego_data['time'].max()
    obs_data = obs_data[(obs_data['time'] >= start_time) & (obs_data['time'] <= end_time)].reset_index(
        drop=True)  # 记录结束时间
    ego_data[['x', 'y']] = ego_data.apply(get_coordinate, axis=1, result_type='expand')
    # ego_data[['x', 'y']] = ego_data.apply(wgs84_to_gcj02, axis=1, result_type='expand')
    init_speed = np.mean(ego_data.loc[0:4, 'speed'].values.tolist())
    ego_data = ego_data[['time', 'x', 'y', 'heading', 'altitude']]
    ego_data['x'] = ego_data['x'] + offset_x
    ego_data['y'] = ego_data['y'] + offset_y

    # plt_trail(ego_data['x'].values.tolist(), ego_data['y'].values.tolist())
    ego_data['date_time'] = pd.to_datetime(ego_data['time'], unit='ms')
    ego_data = ego_data.resample('100ms', on='date_time').mean().bfill()

    ego_data = filter_error(ego_data)

    ego_data['tmp'] = ego_data.index
    time_list = (ego_data.tmp.apply(lambda x: convert(x))).values.tolist()

    if len(ego_data) > 27:
        sos = signal.butter(8, 0.2, 'lowpass', output='sos')
        filtered_x = signal.sosfiltfilt(sos, ego_data['x'])
        filtered_y = signal.sosfiltfilt(sos, ego_data['y'])
        filtered_h = signal.sosfiltfilt(sos, ego_data['heading'])
        ego_data['x'] = signal.savgol_filter(filtered_x, 15, 2, mode='nearest')
        ego_data['y'] = signal.savgol_filter(filtered_y, 15, 2, mode='nearest')
        ego_data['heading'] = signal.savgol_filter(filtered_h, 15, 2, mode='nearest')

    # plt_trail(ego_data['x'].values.tolist(), ego_data['y'].values.tolist())

    obs_data['date_time'] = pd.to_datetime(obs_data['time'], unit='ms')
    # obs_data[['x', 'y']] = obs_data.apply(wgs84_to_gcj02, axis=1, result_type='expand')
    obs_data[['x', 'y']] = obs_data.apply(get_coordinate, axis=1, result_type='expand')
    obs_data['x'] = obs_data['x'] + offset_x
    obs_data['y'] = obs_data['y'] + offset_y
    obs_data['id'] = obs_data['id'].astype('int64')
    obs_data['type'] = obs_data['type'].astype('float')
    obs_data['heading'] = obs_data['heading'].astype('float')
    groups = obs_data.groupby('id')
    obs_list = []

    for obj_id, obs_df in groups:
        if len(obs_df) < 7 or (
                obs_df.iloc[-1]['time'] - obs_df.iloc[-0]['time'] < datetime.timedelta(milliseconds=700)):
            continue

        obs_df = obs_df.reset_index(drop=True)

        # for test
        if obj_id == 3863356330:
            print(110)

        obs_df = filter_error(obs_df)
        obs_df = obs_df.resample('100ms', on='date_time').mean()
        obs_df.dropna(inplace=True, axis=0)
        obs_df['time'] = obs_df.index
        obs_df['time'] = obs_df.time.apply(lambda x: convert(x))
        obs_df = obs_df[['time', 'id', 'type', 'x', 'y', 'altitude', 'heading']]

        # # Cubic Splin
        # spline = Spline(obs_df['x'], obs_df['y'])
        if len(obs_df) > 27:
            sos = signal.butter(8, 0.2, 'lowpass', output='sos')
            filtered_x = signal.sosfiltfilt(sos, obs_df['x'])
            filtered_y = signal.sosfiltfilt(sos, obs_df['y'])
            filtered_h = signal.sosfiltfilt(sos, obs_df['heading'])
            obs_df['x'] = signal.savgol_filter(filtered_x, 15, 2, mode='nearest')
            obs_df['y'] = signal.savgol_filter(filtered_y, 15, 2, mode='nearest')
            obs_df['heading'] = signal.savgol_filter(filtered_h, 15, 2, mode='nearest')
        if len(obs_df) < 7:
            continue
        result_df = pd.merge(obs_df, ego_data, left_index=True, right_index=True, how='left')
        result_df['distance'] = result_df.apply(euclidean_distance, axis=1, result_type='expand')
        distance_count = result_df.loc[result_df['distance'] <= 1.2]

        if len(distance_count) > 3:
            continue
        obs_list.append(obs_df.values.tolist())
    ego_data['tmp'] = pd.to_datetime(ego_data['tmp'])
    ego_data['tmp'] = ego_data['tmp'].astype('int64')
    gps = ego_data.values.tolist()
    ego_position = list()
    for result in gps:
        if len(result) > 0:
            ego_position.append(
                xosc.WorldPosition(x=float(result[1]), y=float(result[2]),
                                   z=float(result[4]) / 10, h=math.radians(float(90 - float(result[3])))))
    return ego_position, obs_list, time_list, init_speed


def load_data_c(ego_path, obs_path):
    ego_data = pd.read_csv(ego_path)

    # 设置origin的原因是时区的时差问题
    ego_data['date_time'] = pd.to_datetime(ego_data['timestampGNSS'], unit='ms', origin='1970-01-01 08:00:00')
    ego_data = ego_data.resample('100ms', on='date_time').mean()
    ego_data['tmp'] = ego_data.index
    time_list = (ego_data.tmp.apply(lambda x: convert(x))).values.tolist()

    init_speed = np.mean(ego_data.head(5)['velocityGNSS'].values.tolist())

    ego_data['time'] = ego_data.index
    ego_data = ego_data[['time', 'longitude', 'latitude', 'heading', 'elevation']]

    ego_data[['x', 'y']] = ego_data.apply(wgs84_to_gcj02, axis=1, result_type='expand')
    # ego_data[['x', 'y']] = ego_data.apply(get_coordinate, axis=1, result_type='expand')
    ego_data['x'] = ego_data['x'] + offset_x
    ego_data['y'] = ego_data['y'] + offset_y
    ego_data['heading'] = -(ego_data['heading'] - 90)
    sos = signal.butter(8, 0.2, 'lowpass', output='sos')
    filtered_x = signal.sosfiltfilt(sos, ego_data['x'])
    filtered_y = signal.sosfiltfilt(sos, ego_data['y'])
    filtered_h = signal.sosfiltfilt(sos, ego_data['heading'])
    ego_data['x'] = signal.savgol_filter(filtered_x, 15, 2, mode='nearest')
    ego_data['y'] = signal.savgol_filter(filtered_y, 15, 2, mode='nearest')
    ego_data['heading'] = signal.savgol_filter(filtered_h, 15, 2, mode='nearest')

    obs_data = pd.read_csv(obs_path)
    obs_data = obs_data[
        ['timestampGNSS', 'typePept', 'ObjectID', 'position_transversal', 'position_longitudinal', 'Xdirection']]
    obs_data.rename(
        columns={'position_transversal': 'x', 'position_longitudinal': 'y', 'ObjectID': 'id', 'Xdirection': 'heading'},
        inplace=True)
    obs_data['date_time'] = pd.to_datetime(obs_data['timestampGNSS'], unit='ms', origin='1970-01-01 08:00:00')
    obs_data['id'] = obs_data['id'].astype('str')
    obs_data['heading'] = obs_data['heading'] + 90
    obs_data['x'] /= 1000
    obs_data['y'] /= 1000

    obs_data = obs_data.dropna()
    groups = obs_data.groupby('id')
    obs_list = []
    for obj_id, obs_df in groups:
        if len(obs_df) < 15:
            continue
        obs_df.insert(loc=len(obs_df.columns), column='z', value=0)
        # obs_df['heading'] = obs_df.speed.apply(lambda x: speed2heading(x))
        # obs_df['v_x'] = obs_df.speed.apply(lambda x: speedx(x))
        # obs_df['v_y'] = obs_df.speed.apply(lambda x: speedy(x))
        # if obs_df['heading'].mean() == 0:
        #     continue

        obs_df = obs_df.resample('100ms', on='date_time').mean()

        obs_df.dropna(inplace=True, axis=0)
        obs_df['time'] = obs_df.index
        obs_df['timestampGNSS'] = obs_df.time.apply(lambda x: convert(x))
        for timestamp, ms_df in obs_df.iterrows():
            tmp_df = ms_df.copy()
            try:
                ego_x = ego_data.loc[timestamp]['x']
                ego_y = ego_data.loc[timestamp]['y']
                heading = math.radians(ego_data.loc[timestamp]['heading'])
                ms_df['x'] = ego_x + (tmp_df['y'] * math.cos(heading) - tmp_df['x'] * math.sin(heading))
                ms_df['y'] = ego_y + (tmp_df['y'] * math.sin(heading) + tmp_df['x'] * math.cos(heading))
                # ms_df['y'] = ego_y + (tmp_df['y'] * math.cos(heading) - tmp_df['x'] * math.cos(heading))
                # ms_df['x'] = ego_x + (tmp_df['y'] * math.sin(heading) + tmp_df['x'] * math.cos(heading))
                ms_df['z'] = ego_data.loc[timestamp]['elevation']
                ms_df['heading'] = ego_data.loc[timestamp]['heading'] + ms_df['heading']
                obs_df.loc[timestamp] = ms_df
            except KeyError:
                break
        obs_list.append(obs_df.values.tolist())
    gps = ego_data.values.tolist()
    ego_position = list()
    for result in gps:
        if len(result) > 0:
            # ego_position.append(
            #     xosc.WorldPosition(x=float(result[5]), y=float(result[6]),
            #                        z=float(result[4]), h=radians(90 - float(result[3]))))
            ego_position.append(
                xosc.WorldPosition(x=float(result[5]), y=float(result[6]),
                                   z=float(result[4]), h=math.radians(float(result[3]))))
    return ego_position, obs_list, time_list, init_speed


def euclidean_distance(row):
    return np.sqrt((row['x_x'] - row['x_y']) ** 2 + (row['y_x'] - row['y_y']) ** 2)


def get_obj_type(mode):
    if mode == WorkMode.roadside.value:
        ped_type = [0]
        car_type = [2, 7]
        bicycle_motor_type = [1, 3]
        bus_type = [5]
    elif mode == WorkMode.car.value:
        ped_type = [2]
        car_type = [6]
        bicycle_motor_type = [2, 3, 4, 5]
        bus_type = [7, 8, 9]
    else:
        ped_type = [7]
        car_type = [2, 3]
        bicycle_motor_type = [8]
        bus_type = []
    return ped_type, car_type, bus_type, bicycle_motor_type


def format_path(input_path, xodr_path="", osgb_path=""):
    """
    data format:
    simulation
        file.xosc
        file.xodr
        file.osgb
    :return:
    """
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if ".xosc" == file[-5:] or ".xml" == file[-4:]:
                for odrFile in os.listdir(root):
                    if xodr_path == "" and ".xodr" == odrFile[-5:]:
                        xodr_path = root + "/" + odrFile
                        break

                for osgbFile in os.listdir(root):
                    if osgb_path == "" and ".osgb" == osgbFile[-5:]:
                        osgb_path = root + "/" + osgbFile
                        break

                path_changer(root + "/" + file, xodr_path, osgb_path)
                print("Change success: " + root + "/" + file)


def changeCDATA(filepath):
    f = open(filepath, "r", encoding="UTF-8")
    txt = f.readline()
    all_line = []
    # txt是否为空可以作为判断文件是否到了末尾
    while txt:
        txt = txt.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"').replace(
            "&apos;", "'")
        all_line.append(txt)
        # 读取文件的下一行
        txt = f.readline()
    f.close()
    f1 = open(filepath, 'w', encoding="UTF-8")
    for line in all_line:
        f1.write(line)
    f1.close()


def generate_osgb(root_path, file):
    VtdRoot = '/home/tang/VIRES/VTD.2021.3'
    ROD = VtdRoot + "/Runtime/Tools/ROD/ROD "
    LicenceAddress = '-s 27500@192.168.11.179'
    RodProject = " --project " + VtdRoot + "/Runtime/Tools/ROD/DefaultProject/DefaultProject.rpj"
    SourceOsgbPath = VtdRoot + "/Runtime/Tools/ROD/DefaultProject/Database"
    xodrFilePath = file
    generate = ROD + LicenceAddress + RodProject + " --xodr " + xodrFilePath + " -G"
    os.system(generate)

    # move osgb file
    sourceOsgbFileName = file[:-4] + "opt.osgb"
    sourceOsgbFilePath = SourceOsgbPath + "/" + sourceOsgbFileName.split('/')[-1]
    destOsgbFilePath = sourceOsgbFileName.replace('opt.osgb', 'osgb')
    os.system("mv " + sourceOsgbFilePath + " " + destOsgbFilePath)

    # remove odr file
    tempXodrFilePath = VtdRoot + "/Runtime/Tools/ROD/DefaultProject/Odr/" + file.split('/')[-1]
    os.system("rm " + tempXodrFilePath)
    print("Complete: " + root_path + "/" + file[:-4] + 'osgb')


def path_changer(xosc_path, xodr_path, osgb_path):
    """
    :param xosc_path:
    :param xodr_path:
    :param osgb_path:
    :return:
    """
    tree = ET.parse(xosc_path)
    treeRoot = tree.getroot()

    # for OpenScenario v0.9, v1.0
    for RoadNetwork in treeRoot.findall('RoadNetwork'):
        for Logics in RoadNetwork.findall('LogicFile'):
            Logics.attrib['filepath'] = xodr_path
        for SceneGraph in RoadNetwork.findall('SceneGraphFile'):
            SceneGraph.attrib['filepath'] = osgb_path

        for Logics in RoadNetwork.findall('Logics'):
            Logics.attrib['filepath'] = xodr_path
        for SceneGraph in RoadNetwork.findall('SceneGraph'):
            SceneGraph.attrib['filepath'] = osgb_path

    # for VTD xml
    for Layout in treeRoot.findall('Layout'):
        Layout.attrib['File'] = xodr_path
        Layout.attrib['Database'] = osgb_path

    tree.write(xosc_path, xml_declaration=True)


# def speed2heading(speed_dict):
#     speed_dict = eval(speed_dict)
#     north_speed = speed_dict['y']
#     east_speed = speed_dict['x']
#     if east_speed == 0:
#         return 0
#     heading = math.degrees(math.atan(north_speed / east_speed))
#     # if east_speed <= 0 <= north_speed:
#     #     heading += 90
#     # elif east_speed <= 0 and north_speed <= 0:
#     #     heading += 180
#     # elif north_speed <= 0 <= east_speed:
#     #     heading -= 90
#     return heading
#
#
# def plt_trail(x, y):
#     # for test
#     min_x = min(x)
#     min_y = min(y)
#     x = (np.array(x) - min_x).tolist()
#     y = (np.array(y) - min_y).tolist()
#     f1 = np.poly1d(np.polyfit(x, y, 4))
#     plt.plot(x, f1(x))
#     plt.plot(x, y, '.')
#     plt.xticks(np.arange(min(x), max(x), 5))
#     plt.yticks(np.arange(min(y), max(y), 5))
#     # plt.xlim(0, 20)  # x坐标轴刻度值范围
#     # plt.ylim(40, 60)  # y坐标轴刻度值范围
#     plt.show()


if __name__ == '__main__':
    # video_list = get_file("/home/tang/Documents/chewang/data/0725output", 'SIMULATION_Camera_1.mp4')
    # for dir_name in video_list:
    #     os.remove(os.path.join(dir_name, 'SIMULATION_Camera_1.mp4'))
    lon = 116.52295885
    lat = 39.79044949
    y, x = transformer.transform(lat, lon)
    print(x, y)
    # generate_osgb('/home/tang/road_model/Osgb', '/home/tang/road_model/Xodr/od_data.xodr')
