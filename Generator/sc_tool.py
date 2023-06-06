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

crs = CRS.from_epsg(4326)

crs_cs = CRS.from_epsg(32650)
transformer = Transformer.from_crs(crs, crs_cs)


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
    def __init__(self, time_stamp=0, ObjectID=0, ObjectType=0, x=0, y=0, z=0, h=0, vel=0):
        self.time = time_stamp
        self.ObjectID = ObjectID
        self.ObjectType = ObjectType
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.h = h
        self.vel = vel


def change_heading(data_df, mode):
    diff = data_df['heading'] - data_df['heading'].shift(1)
    data_df['diff'] = diff.abs()
    data_df['diff'] = data_df['diff'].apply(lambda x: x - 360 if x >= 180 else x)
    diff_data = data_df[abs(data_df['diff']) >= 25]
    if len(diff_data) == 0:
        return data_df
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
    if len(data_df) > 27:
        modified_df = data_df.iloc[0:28, :]
        b, a = signal.butter(8, 0.2, 'lowpass')
        filtered_x = signal.filtfilt(b, a, modified_df['x'])
        filtered_y = signal.filtfilt(b, a, modified_df['y'])
        modified_df['x'] = filtered_x.tolist()

        modified_df['y'] = filtered_y.tolist()
        frames = [modified_df, data_df.iloc[28:, :]]
        data_df = pd.concat(frames)

    data_df['distance'] = ((data_df['x'] - data_df['x'].shift(1)) ** 2 + (
            data_df['y'] - data_df['y'].shift(1)) ** 2) ** 0.5
    data_df = data_df.fillna(0)
    diff = data_df['heading'] - data_df['heading'].shift(1)
    data_df['diff'] = diff.abs()
    diff_data = data_df[abs(data_df['diff']) >= 25]
    if len(diff_data) > 1:
        if (datetime.timedelta(milliseconds=300) < abs(
                diff_data['time'].min() - diff_data['time'].max()) < datetime.timedelta(milliseconds=8000)):
            data_df = change_heading(data_df, mode=0)
        else:
            data_df = change_heading(data_df, mode=1)

    if data_df.loc[1, 'time'] - data_df.loc[0, 'time'] > datetime.timedelta(seconds=4):
        data_df = data_df[1:]

    return data_df


def get_coordinate(x):
    longitude = x['longitude']
    latitude = x['latitude']
    x, y = transformer.transform(latitude, longitude)
    return x, y


def convert(x):
    x = x.to_pydatetime()

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
        # time_now = time.mktime((result[0]))
        time_now = result[0].value / 1000000
        # h = gps[i].h - float(radians(result[2]))
        h = float(math.radians(result[5]))
        z = 0
        position.append(
            ObsPosition((time_now - time_list[0]) / 1000000, str(result[2]), result[1], float(result[3]),
                        float(result[4]), z, h))
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


def load_data_lu(pos_path, target_number_list, target_area, offset_list):
    """
    Load data from single folder(work mode is roadside)
    :param pos_path: the path of file that record objects' position
    :param target_number_list: set of ego plate number
    :param target_area: area number
    :param offset_list: record different area' offset of OpenDrive
    :return:result tuple, include:(ego_position:list, obs_list:list, time_list:list, init_speed:int)
    """
    ego_id_list = list()

    # 新的csv格式数据
    pos_data = pd.read_csv(pos_path)
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
    ego_data = pd.read_csv(os.path.join(os.path.dirname(pos_path), 'ego.csv'))
    ego_data.rename(
        columns={'时间戳': 'datetime', '经度': 'longitude', '纬度': 'latitude', '高程(dm)': 'altitude', '速度(m/s)': 'speed',
                 '航向角(deg)': 'heading'}, inplace=True)
    ego_data['time'] = pd.to_datetime(ego_data['datetime'])

    ego_data['heading'] = ego_data['heading'].astype('float')
    start_time = ego_data['time'].min()
    end_time = ego_data['time'].max()
    obs_data = obs_data[(obs_data['time'] >= start_time) & (obs_data['time'] <= end_time)].reset_index(
        drop=True)  # 记录结束时间
    ego_data[['x', 'y']] = ego_data.apply(get_coordinate, axis=1, result_type='expand')
    init_speed = np.mean(ego_data.loc[0:4, 'speed'].values.tolist())
    ego_data = ego_data[['time', 'x', 'y', 'heading', 'altitude']]
    ego_data['x'] = ego_data['x'] + offset_x
    ego_data['y'] = ego_data['y'] + offset_y

    # plt_trail(ego_data['x'].values.tolist(), ego_data['y'].values.tolist())

    ego_data = filter_error(ego_data)
    ego_data['data_time'] = pd.to_datetime(ego_data['time'], unit='ms')
    ego_data = ego_data.resample('100ms', on='data_time').mean().bfill()

    ego_data['tmp'] = ego_data.index
    time_list = (ego_data.tmp.apply(lambda x: convert(x))).values.tolist()

    # plt_trail(ego_data['x'].values.tolist(), ego_data['y'].values.tolist())

    obs_data['data_time'] = pd.to_datetime(obs_data['time'], unit='ms')
    obs_data[['x', 'y']] = obs_data.apply(get_coordinate, axis=1, result_type='expand')
    obs_data['x'] = obs_data['x'] + offset_x
    obs_data['y'] = obs_data['y'] + offset_y
    obs_data['id'] = obs_data['id'].astype('int64')
    obs_data['type'] = obs_data['type'].astype('float')
    obs_data['heading'] = obs_data['heading'].astype('float')
    groups = obs_data.groupby('id')
    obs_list = []

    for obj_id, obs_df in groups:
        if len(obs_df) < 27:
            continue
        obs_df = obs_df.reset_index(drop=True)

        # # for test
        # if obj_id == 3838623030:
        #     print(110)
        #     print(110)
        # elif obj_id == 3838633430:
        #     print(111)
        #     print(111)
        # elif obj_id == 3838633431:
        #     print(112)
        #     print(112)
        # elif obj_id == 3838633436:
        #     print(113)
        #     print(113)
        # elif obj_id == 3838633566:
        #     print(114)
        #     print(114)
        # print(obj_id)

        obs_df = filter_error(obs_df)
        obs_df = obs_df.resample('100ms', on='data_time').mean()
        obs_df.dropna(inplace=True, axis=0)
        obs_df['time'] = obs_df.index
        obs_df['time'] = obs_df.time.apply(lambda x: convert(x))
        obs_df = obs_df[['time', 'id', 'type', 'x', 'y', 'altitude', 'heading']]
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


def load_data_c(pos_path, obs_path):
    pos_data = pd.read_csv(pos_path)
    # 设置origin的原因是时区的时差问题
    pos_data['data_time'] = pd.to_datetime(pos_data['time'], unit='ms', origin='1970-01-01 08:00:00')
    pos_data = pos_data.resample('100ms', on='data_time').mean()
    pos_data['tmp'] = pos_data.index
    time_list = (pos_data.tmp.apply(lambda x: convert(x))).values.tolist()

    pos_data['time'] = pos_data.index
    pos_data = pos_data[['time', 'longitude', 'latitude', 'heading', 'altitude']]

    base_x, base_y = transformer.transform(pos_data.iloc[0]['longitude'], pos_data.iloc[0]['latitude'])
    pos_data[['x', 'y']] = pos_data.apply(get_coordinate, axis=1, result_type='expand')
    pos_data['x'] = pos_data['x'] - base_x
    pos_data['y'] = pos_data['y'] - base_y
    # x = pos_data['x'].values.tolist()
    # y = pos_data['y'].values.tolist()
    # h = pos_data['heading'].values.tolist()
    # f1 = np.poly1d(np.polyfit(x, y, 4))
    # plt.plot(x, y, '.')
    # plt.plot(x, f1(x))
    # plt.show()

    obs_data = pd.read_csv(obs_path)
    # obs_data = pd.read_excel(obs_path)
    obs_data = obs_data[
        ['time', 'category', 'number', 'position_transversal', 'position_longitudinal', 'Xdirection']]
    obs_data.rename(
        columns={'position_transversal': 'x', 'position_longitudinal': 'y', 'number': 'id', 'X_direction': 'heading'},
        inplace=True)
    obs_data['data_time'] = pd.to_datetime(obs_data['time'], unit='ms', origin='1970-01-01 08:00:00')

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
        obs_df = obs_df.resample('100ms', on='data_time').mean()
        obs_df.dropna(inplace=True, axis=0)
        obs_df['time'] = obs_df.index
        for timestamp, ms_df in obs_df.iterrows():
            tmp_df = ms_df.copy()
            try:
                ego_x = pos_data.loc[timestamp]['x']
                ego_y = pos_data.loc[timestamp]['y']
                heading = math.radians(pos_data.loc[timestamp]['heading'])
                ms_df['x'] = ego_x + tmp_df['x'] * math.cos(heading) + tmp_df['y'] * math.sin(heading)
                ms_df['y'] = ego_y + tmp_df['y'] * math.cos(heading) + tmp_df['x'] * math.sin(heading)
                ms_df['z'] = pos_data.loc[timestamp]['altitude']
                obs_df.loc[timestamp] = ms_df
            except KeyError:
                break

        # x = obs_df['x'].values.tolist()
        # y = obs_df['y'].values.tolist()
        # f1 = np.poly1d(np.polyfit(x, y, 4))
        # plt.axis("equal")
        # plt.plot(x, y, '.')
        # plt.plot(x, f1(x))
        # plt.show()
        obs_list.append(obs_df.values.tolist())
    gps = pos_data.values.tolist()
    ego_position = list()
    for result in gps:
        if len(result) > 0:
            # ego_position.append(
            #     xosc.WorldPosition(x=float(result[5]), y=float(result[6]),
            #                        z=float(result[4]), h=radians(90 - float(result[3]))))
            ego_position.append(
                xosc.WorldPosition(x=float(result[5]), y=float(result[6]),
                                   z=float(result[4]), h=math.radians(float(result[3]))))
    return ego_position, obs_list, time_list


def get_obj_type(mode):
    if mode == WorkMode.roadside.value:
        ped_type = [0]
        car_type = [2, 7]
        bicycle_motor_type = [1, 3]
    elif mode == WorkMode.car.value:
        ped_type = [7]
        car_type = [2, 3]
        bicycle_motor_type = [8]
    else:
        ped_type = [7]
        car_type = [2, 3]
        bicycle_motor_type = [8]
    return ped_type, car_type, bicycle_motor_type


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
    pass
