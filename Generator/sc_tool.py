import math
import time
import datetime
import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from scenariogeneration import xosc
from enum import Enum

# import matplotlib.pyplot as plt

crs = CRS.from_epsg(4326)

crs_cs = CRS.from_epsg(32650)
transformer = Transformer.from_crs(crs, crs_cs)


class Work_Model(Enum):
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


def filter_error(data_df):
    diff = data_df['heading'] - data_df['heading'].shift(1)
    data_df['diff'] = diff.abs()
    diff_data = data_df[abs(data_df['diff']) >= 25]
    if len(diff_data) > 1 and (datetime.timedelta(milliseconds=300) < abs(
            diff_data['time'].min() - diff_data['time'].max()) < datetime.timedelta(milliseconds=5000)):
        error_start = data_df[data_df['time'] == diff_data['time'].min()].index[0] - 1
        error_end = data_df[data_df['time'] == diff_data['time'].max()].index[0] + 1
        # del_time = datetime.timedelta(milliseconds=100)
        error_df = data_df[error_start:error_end]
        _ = error_df[1:-1]
        _['heading'] = np.nan
        error_df[1:-1] = _
        error_df = error_df.interpolate()
        data_df[error_start:error_end] = error_df

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
    time = []
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
        time.append(float(row[1]['Time']))
    return position, x, y, time


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


def smooth_data(csvPath, obsPath):
    data = pd.read_csv(csvPath)
    data = data[data['Type'] != 10]  # 排除非目标物的物体
    data = data.loc[data['ID'] != -1, ['Time', 'ID', 'East', 'North', 'HeadingAngle', 'AbsVel', 'Type']]

    obsdata = pd.read_csv(obsPath)
    obsdata = obsdata[(obsdata['ObjectPosY'] < 5) & (obsdata['ObjectPosY'] > -5) & (obsdata['ObjectPosX'] > -30) & (
            obsdata['ObjectPosX'] < 100)]  # 排除车道线范围外且前向距离较远的目标物
    idlist = obsdata['ObjectID'].tolist()  # 筛选出符合条件的的目标物ID
    data = data[(data['ID'].isin(idlist))]
    for obj_id, obj_data in data.groupby('ID'):
        rating = obj_data['HeadingAngle'] - obj_data['HeadingAngle'].shift(3)
        obj_data['rating'] = rating.abs()
        obj_data.mask(cond=obj_data['rating'] > 1, other=obj_data.shift(3), inplace=True)
        data.loc[data['ID'] == obj_id] = obj_data

    data = data.reset_index(drop=True)

    obsdata = data.copy()
    obsdata = obsdata[['Time', 'ID', 'Type', 'East', 'North', 'HeadingAngle']]
    groups = obsdata.groupby('ID')
    obslist = []
    for key, value in groups:
        obslist.append(value.values.tolist())
        # if len(value) > 15:
        #     obslist.append(value.values.tolist())

    return obslist


def smooth_data_lu(pos_path, target_number, target_area, offset_list):
    ego_id = 0

    # 旧的txt格式数据
    # pos_data = pd.DataFrame(
    #     columns=['time', 'longitude', 'latitude', 'heading', 'altitude', 'type', 'id', 'speed'])
    # with open(file=pos_path, encoding='utf8') as f:
    #     base_data = f.readlines()
    # for ms in base_data:
    #     ms_dict = eval(ms)
    #     for obj in ms_dict['targets']:
    #         if (obj['type']) != 2 and (obj['type']) != 0 and (obj['type']) != 7:
    #             continue
    #         tmp_df = pd.DataFrame(
    #             columns=['time', 'longitude', 'latitude', 'heading', 'altitude', 'type', 'id', 'speed'])
    #         tmp_df.loc[0, 'time'] = ms_dict['timestamp']
    #         tmp_df.loc[0, 'longitude'] = obj['longitude']
    #         tmp_df.loc[0, 'latitude'] = obj['latitude']
    #         tmp_df.loc[0, 'heading'] = obj['heading']
    #         tmp_df.loc[0, 'altitude'] = obj['elevation']
    #         tmp_df.loc[0, 'type'] = obj['type']
    #         tmp_df.loc[0, 'speed'] = obj['speed']
    #         tmp_df.loc[0, 'id'] = obj['uuid'][-8:]
    #         if ego_id == 0 and obj['plateNo'] in plate_list and obj['plateNo'] == target_number:
    #             ego_id = obj['uuid'][-8:]
    #         pos_data = pd.concat([pos_data, tmp_df])

    # 新的txt格式数据
    pos_data = pd.read_csv(pos_path)
    pos_data.rename(
        columns={'时间戳': 'datetime', '感知目标ID': 'id', '感知目标经度': 'longitude', '感知目标纬度': 'latitude',
                 '高程(dm)': 'altitude', '速度(m/s)': 'speed', '航向角(deg)': 'heading', '感知目标类型': 'type'},
        inplace=True)
    pos_data['datetime'] = pd.to_datetime(pos_data['datetime'])
    pos_data['time'] = pos_data['datetime'].astype('int64')
    pos_data['time'] = pd.to_datetime(pos_data['datetime'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
        'Asia/Shanghai')
    pos_data = pos_data[['time', 'id', 'type', 'longitude', 'latitude', 'speed', 'heading', 'altitude']]
    pos_data['id'] = pos_data['id'].astype('str')
    pos_data.id = pos_data.id.apply(lambda x: x[-10:])
    if target_number in pos_data['id'].values:
        ego_id = target_number

    offset_x, offset_y = (-1, -1)
    for offset in offset_list:
        if target_area == offset[4:7]:
            offset_value = offset.split(':')[1]
            offset_x = float(offset_value.split(',')[0])
            offset_y = float(offset_value.split(',')[1])
            break
    if ego_id == 0:
        return 401
    elif offset_x == -1 and offset_y == -1:
        return 402
    pos_data = pos_data.reset_index(drop=True)
    ego_data = pos_data[pos_data['id'] == ego_id].reset_index(drop=True)
    obs_data = pos_data[pos_data['id'] != ego_id].reset_index(drop=True)
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
    # 设置origin的原因是时区的时差问题
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


def smooth_data_c(pos_path, obs_path):
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
    obslist = []
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
        obslist.append(obs_df.values.tolist())
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
    return ego_position, obslist, time_list


def get_obj_type(model):
    if model == Work_Model.roadside:
        ped_type = [0]
        car_type = [2, 7]
        bicycle_motor_type = [1, 3]
    elif model == Work_Model.car:
        ped_type = [7]
        car_type = [2, 3]
        bicycle_motor_type = [8]
    else:
        ped_type = [7]
        car_type = [2, 3]
        bicycle_motor_type = [8]
    return ped_type, car_type, bicycle_motor_type


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
