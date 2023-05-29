import numpy as np
import pandas as pd
from time import mktime
from datetime import timedelta
from pyproj import CRS, Transformer
from math import radians
from scenariogeneration import xosc

# import matplotlib.pyplot as plt


crs = CRS.from_epsg(4326)

crs_cs = CRS.from_epsg(32650)
transformer = Transformer.from_crs(crs, crs_cs)


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
    def __init__(self, time=0, ObjectID=0, ObjectType=0, x=0, y=0, z=0, h=0, vel=0):
        self.time = time
        self.ObjectID = ObjectID
        self.ObjectType = ObjectType
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.h = h
        self.vel = vel


def read_gps(obsList, time_list):
    position = []
    for result in obsList:
        time_now = (result[0] - time_list[0]) / 1000
        h = radians(float(90 - result[6]))
        z = result[5] / 10
        position.append(
            ObsPosition(time_now, str(result[1]), result[2], float(result[3]), float(result[4]), z, h))
    return position


def convert(x):
    x = x.to_pydatetime()

    timeStamp = int(mktime(x.timetuple()) * 1000.0 + x.microsecond / 1000.0)

    return timeStamp


def speedx(speed_dict):
    speed_dict = eval(speed_dict)
    east_speed = speed_dict['x']
    if east_speed == 0:
        return 0
    return east_speed


def speedy(speed_dict):
    speed_dict = eval(speed_dict)
    east_speed = speed_dict['x']
    north_speed = speed_dict['y']
    if east_speed == 0:
        return 0
    return north_speed


# def plt_trail(x, y):
#     min_x = min(x)
#     min_y = min(y)
#     x = (np.array(x)-min_x).tolist()
#     y = (np.array(y)-min_y).tolist()
#     f1 = np.poly1d(np.polyfit(x, y, 4))
#     plt.plot(x, f1(x))
#     plt.plot(x, y, '.')
#     plt.xticks(np.arange(min(x), max(x), 5))
#     plt.yticks(np.arange(min(y), max(y), 5))
#     # plt.xlim(0, 20)  # x坐标轴刻度值范围
#     # plt.ylim(40, 60)  # y坐标轴刻度值范围
#     plt.show()

def filter_error(data_df):
    diff = data_df['heading'] - data_df['heading'].shift(1)
    data_df['diff'] = diff.abs()
    diff_data = data_df[abs(data_df['diff']) >= 25]
    if len(diff_data) > 1 and (timedelta(milliseconds=300) < abs(
            diff_data['time'].min() - diff_data['time'].max()) < timedelta(milliseconds=5000)):
        error_start = data_df[data_df['time'] == diff_data['time'].min()].index[0] - 1
        error_end = data_df[data_df['time'] == diff_data['time'].max()].index[0] + 1
        # del_time = datetime.timedelta(milliseconds=100)
        error_df = data_df[error_start:error_end]
        _ = error_df[1:-1]
        _['heading'] = np.nan
        error_df[1:-1] = _
        error_df = error_df.interpolate()
        data_df[error_start:error_end] = error_df

    if data_df.loc[1, 'time'] - data_df.loc[0, 'time'] > timedelta(seconds=4):
        data_df = data_df[1:]

    return data_df


def smooth_data(pos_path, target_number, target_area, offset_list):
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
                                   z=float(result[4]) / 10, h=radians(float(90 - float(result[3])))))
    return ego_position, obs_list, time_list, init_speed


def get_coordinate(x):
    longitude = x['longitude']
    latitude = x['latitude']
    x, y = transformer.transform(latitude, longitude)
    return x, y


if __name__ == '__main__':
    pass
