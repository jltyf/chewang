import math
import time

import pyproj
from pyproj import CRS, Transformer
import pandas as pd
from scenariogeneration import xosc


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


def abs(x):
    if x >= 0:
        return x
    else:
        return -x


def read_obs(obsList, time):
    position = []

    for result in obsList:

        if len(result) > 0:
            position.append(
                ObsPosition((float(result[0]) - time[0]) / 1000, str(result[1]), result[2], result[3],
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


def convert(x):
    x = x.to_pydatetime()

    timeStamp = int(time.mktime(x.timetuple()) * 1000.0 + x.microsecond / 1000.0)

    return timeStamp


# def change_heading(heading):
#     if heading
#
#     return timeStamp


def speed2heading(speed_dict):
    speed_dict = eval(speed_dict)
    north_speed = speed_dict['y']
    east_speed = speed_dict['x']
    if east_speed == 0:
        return 0
    heading = math.degrees(math.atan(north_speed / east_speed))
    # if east_speed <= 0 <= north_speed:
    #     heading += 90
    # elif east_speed <= 0 and north_speed <= 0:
    #     heading += 180
    # elif north_speed <= 0 <= east_speed:
    #     heading -= 90
    return heading


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


def change_CDATA(filepath):
    '行人场景特例，对xosc文件内的特殊字符做转换'
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


def get_coordinate_new_2(x):
    longitude = x['longitude']
    latitude = x['latitude']
    crs = CRS.from_epsg(4326)

    crs_cs = pyproj.CRS.from_epsg(32650)
    transformer = Transformer.from_crs(crs, crs_cs)
    x, y = transformer.transform(latitude, longitude)
    return x, y


if __name__ == '__main__':
    pass
