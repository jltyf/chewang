from time import mktime
from pyproj import CRS, Transformer
from math import radians, sin, cos, degrees, atan
from scenariogeneration import xosc
import pandas as pd

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
        # time_now = time.mktime((result[0]))
        time_now = result[0].value / 1000000
        # h = gps[i].h - float(radians(result[2]))
        h = float(radians(result[5]))
        z = 0
        position.append(
            ObsPosition((time_now - time_list[0]) / 1000000, str(result[2]), result[1], float(result[3]),
                        float(result[4]), z, h))
    return position


def convert(x):
    x = x.to_pydatetime()

    timeStamp = int(mktime(x.timetuple()) * 1000.0 + x.microsecond / 1000.0)

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
    heading = degrees(atan(north_speed / east_speed))
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


def smooth_data(pos_path, obs_path):
    pos_data = pd.read_csv(pos_path)
    # 设置origin的原因是时区的时差问题
    pos_data['data_time'] = pd.to_datetime(pos_data['time'], unit='ms', origin='1970-01-01 08:00:00')
    pos_data = pos_data.resample('100ms', on='data_time').mean()
    pos_data['tmp'] = pos_data.index
    time_list = (pos_data.tmp.apply(lambda x: convert(x))).values.tolist()

    pos_data['time'] = pos_data.index
    pos_data = pos_data[['time', 'longitude', 'latitude', 'heading', 'altitude']]

    base_x, base_y = get_coordinate_new(pos_data.iloc[0]['longitude'], pos_data.iloc[0]['latitude'])
    pos_data[['x', 'y']] = pos_data.apply(get_coordinate_new_2, axis=1, result_type='expand')
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
                heading = radians(pos_data.loc[timestamp]['heading'])
                ms_df['x'] = ego_x + tmp_df['x'] * cos(heading) + tmp_df['y'] * sin(heading)
                ms_df['y'] = ego_y + tmp_df['y'] * cos(heading) + tmp_df['x'] * sin(heading)
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
                                   z=float(result[4]), h=radians(float(result[3]))))
    return ego_position, obslist, time_list


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


def get_coordinate(longitude, latitude):
    crs = CRS.from_epsg(4326)

    crs_cs = CRS.from_epsg(32650)
    transformer = Transformer.from_crs(crs, crs_cs)
    x, y = transformer.transform(latitude, longitude)
    return x - 455813.908131, y - 4401570.684274


def get_coordinate_new(longitude, latitude):
    crs = CRS.from_epsg(4326)

    crs_cs = CRS.from_epsg(32650)
    transformer = Transformer.from_crs(crs, crs_cs)
    x, y = transformer.transform(latitude, longitude)
    return x, y


def get_coordinate_new_2(x):
    longitude = x['longitude']
    latitude = x['latitude']
    crs = CRS.from_epsg(4326)

    crs_cs = CRS.from_epsg(32650)
    transformer = Transformer.from_crs(crs, crs_cs)
    x, y = transformer.transform(latitude, longitude)
    return x, y


def cal_pos(x):
    ego_x = x['ego_x']
    ego_y = x['ego_y']
    obj_x = x['obj_x']
    obj_y = x['obj_y']
    heading = radians(x['heading'])
    real_x = ego_x + obj_x * cos(heading) + obj_y * sin(heading)
    real_y = ego_y + obj_y * cos(heading) + obj_x * sin(heading)
    return real_x, real_y


# def transform_coordinate(lon, lat):
#     url = 'https://restapi.amap.com/v3/assistant/coordinate/convert?parameters'
#     key = 'cdf24f471cc579ba6d5dd1f9b856ee31'
#     params = {
#         'key': key,
#         'locations': f'{lon},{lat}',
#         'coordsys': 'gps'
#     }
#     res_coor = (loads(requests.get(url=url, params=params).text))['locations'].split(',')
#     return res_coor[0], res_coor[1]


if __name__ == '__main__':
    a = get_coordinate_new(116.49029609, 39.7621247)
    print(a)
