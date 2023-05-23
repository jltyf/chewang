import json
import math
import os
import traceback
import warnings

import xml.etree.ElementTree as ET
from datetime import datetime

from PyQt5.QtWidgets import QApplication
from scenariogeneration import ScenarioGenerator
from scenariogeneration import xodr
from scenariogeneration import xosc

from sc_tool_lu import read_gps, smooth_data

warnings.filterwarnings("ignore")


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
    provided by Dongpeng Ding
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


class ScenarioModel(object):
    def __init__(self):
        pbb = xosc.BoundingBox(0.5, 0.5, 1.8, 2.0, 0, 0.9)
        mbb = xosc.BoundingBox(1, 2, 1.7, 1.5, 0, 0.9)
        bb = xosc.BoundingBox(2.1, 4.5, 1.8, 1.5, 0, 0.9)
        fa = xosc.Axle(0.5, 0.6, 1.8, 3.1, 0.3)
        ba = xosc.Axle(0, 0.6, 1.8, 0, 0.3)
        prop = xosc.Properties()
        self.catalog = xosc.Catalog()
        self.catalog.add_catalog('VehicleCatalog', 'Distros/Current/Config/Players/Vehicles')
        self.catalog.add_catalog('PedestrianCatalog', 'Distros/Current/Config/Players/Pedestrians')
        self.catalog.add_catalog('ControllerCatalog', 'Distros/Current/Config/Players/driverCfg.xml')
        self.red_veh = xosc.Vehicle('Audi_A3_2009_black', xosc.VehicleCategory.car, bb, fa, ba, 69.444, 200, 10)
        self.red_veh.add_property(name='control', value='external')
        self.white_veh = xosc.Vehicle('Audi_A3_2009_red', xosc.VehicleCategory.car, bb, fa, ba, 69.444, 200, 10)
        self.male_ped = xosc.Pedestrian('Christian', 'male_adult', 70, xosc.PedestrianCategory.pedestrian, pbb)
        self.motorcycle = xosc.Vehicle('Kawasaki_ZX-9R_black', xosc.VehicleCategory.motorbike, mbb, fa, ba, 69.444, 200,
                                       10)
        self.cnt = xosc.Controller('DefaultDriver', prop)
        self.cnt2 = xosc.Controller('No Driver', prop)
        self.ego_name = 'Ego'
        self.obj_name = 'Player'
        self.entities = xosc.Entities()


class Scenario(ScenarioGenerator):
    def __init__(self, gps, obs, gpsTime, sceperiod, ped_flag, abspath, init_speed):
        ScenarioGenerator.__init__(self)
        self.gps = gps
        self.obs = obs
        self.gpsTime = gpsTime
        self.ObjectID = 0
        self.egoSpeed = 5  # 自车初始速度，目前无用
        self.sceperiod = sceperiod  # 场景持续的时间
        self.weather = xosc.PrecipitationType.dry  # 默认是晴天
        self.visual_fog_range = 20000  # 正常的能见度，没有雾
        self.time = (True, 2019, 12, 19, 12, 0, 0)  # 用于判断白天还是黑夜
        self.cloudstate = xosc.CloudState.free
        self.pedestrian_flag = ped_flag
        self.object_dict = {}
        self.abspath = abspath
        self.init_speed = init_speed

    def road(self):
        positionEgo = self.gps
        planview = xodr.PlanView()

        for i in range(len(positionEgo) - 1):
            x = 0.000001 if abs(positionEgo[i].x) < 0.000001 else positionEgo[i].x
            y = 0.000001 if abs(positionEgo[i].y) < 0.000001 else positionEgo[i].y
            nextx = 0.000001 if abs(positionEgo[i + 1].x) < 0.000001 else positionEgo[i + 1].x
            nexty = 0.000001 if abs(positionEgo[i + 1].y) < 0.000001 else positionEgo[i + 1].y
            h = float(positionEgo[i].h)

            planview.add_fixed_geometry(xodr.Line(math.sqrt(math.pow(nextx - x, 2) + math.pow(nexty - y, 2))), x, y, h)

        planview.add_fixed_geometry(xodr.Line(100), nextx, nexty, h)
        # create two different roadmarkings
        rm_solid = xodr.RoadMark(xodr.RoadMarkType.solid, 0.2)
        rm_dashed = xodr.RoadMark(xodr.RoadMarkType.broken, 0.2)

        # create simple lanes
        lanes = xodr.Lanes()
        lanesection1 = xodr.LaneSection(0, xodr.standard_lane(offset=1.75, rm=rm_dashed))
        lanesection1.add_left_lane(xodr.standard_lane(offset=3.5, rm=rm_dashed))
        lanesection1.add_left_lane(xodr.standard_lane(offset=3.5, rm=rm_dashed))
        lanesection1.add_left_lane(xodr.standard_lane(offset=3.5, rm=rm_solid))
        lanesection1.add_right_lane(xodr.standard_lane(offset=3.5, rm=rm_dashed))
        lanesection1.add_right_lane(xodr.standard_lane(offset=3.5, rm=rm_dashed))
        lanesection1.add_right_lane(xodr.standard_lane(offset=3.5, rm=rm_solid))
        lanes.add_lanesection(lanesection1)
        lanes.add_laneoffset(xodr.LaneOffset(a=1.75))

        road = xodr.Road(0, planview, lanes)
        odr = xodr.OpenDrive('myroad')
        odr.add_road(road)
        odr.adjust_roads_and_lanes()
        return odr

    def getH(self, point1, point2):
        h = math.atan2((point2.y - point1.y), (point2.x - point1.x))
        return h

    def scenario(self, **kwargs):
        road = xosc.RoadNetwork(self.road_file, scenegraph="/simulation0.osgb")
        scenario_model = ScenarioModel()
        scenario_model.entities.add_scenario_object(scenario_model.ego_name, scenario_model.red_veh, scenario_model.cnt)
        positionEgo = self.gps
        obj_count = 0
        for i in range(len(self.obs)):
            object_list = []
            for j in self.obs[i]:
                object_list.append(j.ObjectType)
            self.obs[i][0].ObjectType = max(object_list, key=object_list.count)
            if self.pedestrian_flag == 1 and self.obs[i][0].ObjectType == 0:
                self.object_dict[scenario_model.obj_name + str(obj_count)] = self.obs[i]
                scenario_model.entities.add_scenario_object(scenario_model.obj_name + str(i), scenario_model.male_ped)
            elif self.obs[i][0].ObjectType == 2 or self.obs[i][0].ObjectType == 7:
                self.object_dict[scenario_model.obj_name + str(obj_count)] = self.obs[i]
                scenario_model.entities.add_scenario_object(scenario_model.obj_name + str(i), scenario_model.white_veh,
                                                            scenario_model.cnt2)
            elif self.obs[i][0].ObjectType == 1 or self.obs[i][0].ObjectType == 3:
                self.object_dict[scenario_model.obj_name + str(obj_count)] = self.obs[i]
                scenario_model.entities.add_scenario_object(scenario_model.obj_name + str(i), scenario_model.motorcycle,
                                                            scenario_model.cnt2)
            obj_count += 1

        init = xosc.Init()
        step_time = xosc.TransitionDynamics(xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 1)
        egospeed = xosc.AbsoluteSpeedAction(self.init_speed, step_time)
        objspeed = xosc.AbsoluteSpeedAction(0, step_time)

        # init
        step = len(positionEgo) / self.sceperiod
        step_dataEgo = []
        positionEgo1 = []
        init.add_init_action(scenario_model.ego_name, xosc.TeleportAction(
            xosc.WorldPosition(x=positionEgo[0].x, y=positionEgo[0].y, z=positionEgo[0].z, h=positionEgo[0].h, p=0,
                               r=0)))
        init.add_init_action(scenario_model.ego_name, egospeed)

        # ego car
        trajectory = xosc.Trajectory('oscTrajectory0', False)
        lasth = float(positionEgo[0].h)
        for j in range(len(positionEgo)):
            if j == 0:
                # time = 15.01
                time = 0
            else:
                time = round(((self.gpsTime[j] - self.gpsTime[0]) / 1000), 2)
                # time = round(((self.gpsTime[j] - self.gpsTime[0]) / 1000000000), 2)

            x = float(positionEgo[j].x)
            y = float(positionEgo[j].y)
            z = float(positionEgo[j].z)

            if (j > 0) & (float(positionEgo[j].h - lasth) < -6):
                h = float(positionEgo[j].h) + 2 * math.pi
            elif (j > 0) & (float(positionEgo[j].h - lasth) > 6):
                h = float(positionEgo[j].h) - 2 * math.pi
            else:
                h = float(positionEgo[j].h)
            if h == 0:
                h = 0.000001
            step_dataEgo.append(time)
            # positionEgo1.append(xosc.WorldPosition(x=x, y=y, z=z, h=h, p=0, r=0))
            positionEgo1.append(xosc.WorldPosition(x=x, y=y, z=0, h=h, p=0, r=0))
            lasth = h

        true_end_time = step_dataEgo[-1]

        for _ in range(math.ceil(step)):
            step_dataEgo.append(true_end_time + round(_ / step, 2))
            positionEgo1.append(
                xosc.WorldPosition(x=positionEgo[-1].x, y=positionEgo[-1].y, h=positionEgo[-1].h,
                                   p=0, r=0))
        polyline = xosc.Polyline(step_dataEgo, positionEgo1)
        trajectory.add_shape(polyline)

        speedaction = xosc.FollowTrajectoryAction(trajectory, xosc.FollowMode.position, xosc.ReferenceContext.absolute,
                                                  1, 0)
        trigger = xosc.ValueTrigger(name='drive_start_trigger', delay=0, conditionedge=xosc.ConditionEdge.rising,
                                    valuecondition=xosc.SimulationTimeCondition(value=0, rule=xosc.Rule.greaterThan))

        event = xosc.Event('Event1', xosc.Priority.overwrite)
        event.add_trigger(trigger)
        event.add_action('newspeed', speedaction)
        man = xosc.Maneuver('my maneuver')
        man.add_event(event)

        mangr = xosc.ManeuverGroup('mangroup', selecttriggeringentities=True)
        mangr.add_actor('Ego')
        mangr.add_maneuver(man)

        trigger0 = xosc.Trigger('start')
        act = xosc.Act('Act1', trigger0)
        act.add_maneuver_group(mangr)

        story1 = xosc.Story('mystory_ego')
        story1.add_act(act)

        sb = xosc.StoryBoard(init, stoptrigger=xosc.ValueTrigger('stop_trigger', 0, xosc.ConditionEdge.none,
                                                                 xosc.SimulationTimeCondition(step_dataEgo[-1],
                                                                                              xosc.Rule.greaterThan),
                                                                 'stop'))
        sb.add_story(story1)

        # obj
        for player in self.object_dict:
            if not self.object_dict:
                break
            row = self.object_dict[player]
            name = player
            positionM = []
            step_dataM = []
            rowNew = row
            lasth = float(rowNew[0].h)

            for j in range(len(rowNew) - 1):
                x = float(rowNew[j].x)
                y = float(rowNew[j].y)
                z = float(rowNew[j].z)
                h = float(rowNew[j].h)

                positionM.append(
                    # xosc.WorldPosition(x=x, y=y, z=z, h=h, p=0, r=0))
                    xosc.WorldPosition(x=x, y=y, z=0, h=h, p=0, r=0))
                step_dataM.append(float(rowNew[j].time))
                lasth = h
            first_time = rowNew[0].time
            last_time = rowNew[-1].time

            add_action = xosc.DeleteEntityAction(name)
            add_trigger = xosc.ValueTrigger(name='entity_add_trigger', delay=0,
                                            conditionedge=xosc.ConditionEdge.rising,
                                            valuecondition=xosc.SimulationTimeCondition(value=first_time,
                                                                                        rule=xosc.Rule.greaterThan))
            add_event = xosc.Event('Event_del', xosc.Priority.overwrite)
            add_event.add_trigger(add_trigger)
            add_event.add_action('del_action', add_action)
            man.add_event(add_event)

            trajectoryM = xosc.Trajectory('oscTrajectory1', False)
            polylineM = xosc.Polyline(step_dataM, positionM)
            trajectoryM.add_shape(polylineM)

            speedaction2 = xosc.FollowTrajectoryAction(trajectoryM, xosc.FollowMode.position,
                                                       xosc.ReferenceContext.absolute, 1, 0)

            event2 = xosc.Event('Event1', xosc.Priority.overwrite)
            trigger2 = xosc.EntityTrigger("obj_start_trigger", 0, xosc.ConditionEdge.rising,
                                          xosc.SpeedCondition(0, xosc.Rule.greaterThan), 'Ego')
            event2.add_trigger(trigger2)

            if self.pedestrian_flag and row[0].ObjectType == 0:
                pedaction = xosc.FollowTrajectoryAction(trajectoryM, xosc.FollowMode.position,
                                                        xosc.ReferenceContext.absolute, 1, 0)
                event2.add_action('newspeed', pedaction)

            else:
                event2.add_action('newspeed', speedaction2)

            man = xosc.Maneuver('my maneuver')
            man.add_event(event2)

            event3 = xosc.Event('Event_ped', xosc.Priority.overwrite)
            event3.add_trigger(trigger)
            del_action = xosc.DeleteEntityAction(name)
            del_trigger = xosc.ValueTrigger(name='entity_del_trigger', delay=0,
                                            conditionedge=xosc.ConditionEdge.rising,
                                            valuecondition=xosc.SimulationTimeCondition(value=last_time,
                                                                                        rule=xosc.Rule.greaterThan))
            del_event = xosc.Event('Event_del', xosc.Priority.overwrite)
            del_event.add_trigger(del_trigger)
            del_event.add_action('del_action', del_action)
            man.add_event(del_event)

            if self.pedestrian_flag and row[0].ObjectType == 0:
                action3 = xosc.CustomCommandAction(0, 0, 0, 0, 1, 0, 0)
                speed = 3
                motion = 'walk'
                text = f'<Traffic>\n     <ActionMotion actor="{name}" move="{motion}" speed="{speed}" force="false" rate="0" delayTime="0.0" activateOnExit="false"/>\n</Traffic>'
                newnode = ET.Element("CustomCommandAction")
                newnode.attrib = {'type': 'scp'}
                newnode.text = f"<![CDATA[\n{text}]\n]>"
                action3.add_element(newnode)
                event3.add_action('newspeed', action3)
                man.add_event(event3)

            mangr2 = xosc.ManeuverGroup('mangroup', selecttriggeringentities=True)
            mangr2.add_actor(name)
            mangr2.add_maneuver(man)

            act2 = xosc.Act('Act1', trigger0)
            act2.add_maneuver_group(mangr2)

            story2 = xosc.Story('mystory_' + name)
            story2.add_act(act2)

            sb.add_story(story2)
        paramet = xosc.ParameterDeclarations()
        sce = xosc.Scenario('my scenario', 'Maggie', paramet, scenario_model.entities, sb, road, scenario_model.catalog)
        return sce


class Task_lu:
    def __init__(self, path, keyFileName):
        """"初始化方法"""
        self.path = path
        self.keyFileName = keyFileName

    def getFile(self, path, keyFileName):
        '''
        获得可以进行轨迹提取的所有片段文件夹路径

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        FileList : TYPE
            可以进行轨迹提取的所有片段文件夹路径.

        '''
        files = os.listdir(path)  # 得到文件夹下的所有文件，包含文件夹名称

        FileList = []
        if keyFileName not in files:
            for name in files:
                if os.path.isdir(path + '/' + name):
                    FileList.extend(self.getFile(path + '/' + name, keyFileName))  # 回调函数，对所有子文件夹进行搜索
        else:
            FileList.append(path)
        FileList = list(set(FileList))

        return FileList

    def format_two(self, input_path):
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

                    xodrFilePath = ""
                    osgbFilePath = ""

                    for odrFile in os.listdir(root):
                        if ".xodr" == odrFile[-5:]:
                            xodrFilePath = root + "/" + odrFile
                            break

                    for osgbFile in os.listdir(root):
                        if ".osgb" == osgbFile[-5:]:
                            osgbFilePath = root + "/" + osgbFile
                            break

                    path_changer(root + "/" + file, xodrFilePath, osgbFilePath)
                    print("Change success: " + root + "/" + file)

    def changeCDATA(self, filepath):
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

    def generateScenarios_test(self, abs_path, output):
        pos_path = os.path.join(abs_path, 'data.csv')
        information_path = os.path.join(abs_path, 'information.json')
        with open(file='offset.txt', encoding='utf8') as f:
            offset_list = f.read().splitlines()

        if abs_path != output:
            output = os.path.join(output, os.path.basename(abs_path))
            if not os.path.exists(output):
                os.makedirs(output)
        with open(information_path, encoding='utf-8') as f:
            file_contents = f.read()
        parsed_json = json.loads(file_contents, encoding='utf-8')
        target_number = parsed_json['number']
        target_area = parsed_json['area']
        results = smooth_data(pos_path, target_number, target_area, offset_list)
        gps, obs_list, time, init_speed = results[0], results[1], results[2], results[3]
        obsL = []
        for obj in obs_list:
            if len(obj) > 10:
                obsL.append(read_gps(obj, time))

        sceperiod = math.ceil((time[-1] - time[0]) / 1000)
        ped_flag = True

        s = Scenario(gps, obsL, time, sceperiod, ped_flag, abs_path, init_speed)
        s.print_permutations()
        filename = output + '/SIMULATION'
        if not os.path.exists:
            os.makedirs(filename)
        files = s.generate(filename)
        generate_osgb(output_path, files[0][0].replace('xosc', 'xodr'))
        self.format_two(filename)
        self.changeCDATA(files[0][0])

    def generateScenarios(self, abs_path, output, textBrowser=0):
        pos_path = os.path.join(abs_path, 'data.csv')
        information_path = os.path.join(abs_path, 'information.json')
        with open(file='offset.txt', encoding='utf8') as f:
            offset_list = f.read().splitlines()

        with open(information_path, encoding='utf-8') as f:
            file_contents = f.read()

        if abs_path != output:
            output = os.path.join(output, os.path.basename(abs_path))
            if not os.path.exists(output):
                os.makedirs(output)

        parsed_json = json.loads(file_contents, encoding='utf-8')
        target_number = parsed_json['number']
        target_area = parsed_json['area']
        results = smooth_data(pos_path, target_number, target_area, offset_list)
        if results == 401:
            textBrowser.append(f'场景片段{abs_path}未找到自动驾驶车辆')
            QApplication.processEvents()
        elif results == 402:
            textBrowser.append(f'场景片段{abs_path}未找到对应区域')
            QApplication.processEvents()
        else:
            gps, obs_list, time, init_speed = results[0], results[1], results[2], results[3]
        obsL = []
        for obj in obs_list:
            if len(obj) > 10:
                obsL.append(read_gps(obj, time))

        sceperiod = math.ceil((time[-1] - time[0]) / 1000)
        ped_flag = True

        s = Scenario(gps, obsL, time, sceperiod, ped_flag, abs_path, init_speed)
        s.print_permutations()
        filename = output + '/SIMULATION'
        if not os.path.exists:
            os.makedirs(filename)
        files = s.generate(filename)
        self.changeCDATA(files[0][0])
        textBrowser.append(f'场景{output}还原成功')
        QApplication.processEvents()

    def batchRun(self, input_path, output, textBrowser):
        files = self.getFile(input_path, self.keyFileName)
        error_count = 0
        correct_count = 0
        if len(files) == 0:
            textBrowser.clear()
            textBrowser.append('输入路径中未包含场景数据文件，请确认路径输入是否正确')
            QApplication.processEvents()
            return
        for di, absPath in enumerate(sorted(files)):
            QApplication.processEvents()
            try:
                self.generateScenarios(absPath, output, textBrowser)
                correct_count += 1
            except:
                textBrowser.append(f'场景{absPath}还原失败')
                QApplication.processEvents()
                error_count += 1
                error = {'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                         'traceback': traceback.format_exc()}
                with open('error.log', 'a+') as f:
                    json.dump(error, f, indent=4)
                    f.write('\n')
            finally:
                textBrowser.moveCursor(textBrowser.textCursor().End)
        textBrowser.clear()
        textBrowser.append(f'本次任务运行完成')
        textBrowser.append(f'本次任务中成功还原场景数：{correct_count}\n本次任务中未能还原场景数：{error_count}')
        textBrowser.moveCursor(textBrowser.textCursor().End)

    def batchRun_test(self, input_path, output):
        files = self.getFile(input_path, self.keyFileName)
        for di, absPath in enumerate(sorted(files)):
            self.generateScenarios_test(absPath, output)


if __name__ == "__main__":
    rootPath = "/home/tang/Documents/chewang/csvdata"
    output_path = "/home/tang/Documents/chewang/0523"
    a = Task_lu(rootPath, "data.csv")

    # 生成场景
    a.batchRun_test(rootPath, output_path)
