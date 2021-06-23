# AgentManagerment, batch
import sys
# sys.path.append('../')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from GoStatus import GlobalStatus, LocalFullStatus
from GoConfig import GoConfig
from multiprocessing import Manager, Condition
# from threading import Condition
import math
import networkx as nx
import time
from numba import jit
import numba as nb


'''
notes:
    1. create and delete ProcessAgent with same map(same of roads and allow-areas)
    2. sharing status of agents.
    3. dispatch tasks.(start-pos / goal-pos)
'''


class GoSharedStatus:
    def __init__(self):
        print("GoSharedStatus------------------------->")
        self.ev_path = None
        self.map_pic_str = None
        self.roads_pic_url = None
        self.allow_area_pic_url = None
        self.one_channel_map_pic_str = None
        self.name = None
        self.allow_area_gunplot_cfg_url = None
        self.roads_gunplot_cfg_url = None
        self.staic_map_array = None
        self.map_width = 0
        self.map_height = 0
        self.agents_dynamic_status = Manager().dict()
        self.agents_con = Manager().Condition()
        self.G = nx.Graph()
        self.road_node_Nos =None
        self.road_node_info = None
        self.road_lines = None
        self.road_lines_num = None
        self.node_edges = None
        self.target_range = 0
        # process environment
        # self.init_map()

    def get_self_current_status(self, myid=-1):
        return self.agents_dynamic_status.get(myid)

    def get_other_agents_status(self, myid=-1):
        results = []
        # self_status = list()
        # for (k, v) in self.agents_dynamic_status.items():
        #     # print("K", k)
        #     if k == myid:
        #         # self_status.append(v)
        #         continue
        #     results.append(v)
        for k in list(self.agents_dynamic_status.keys()):
            if k == myid:
                continue
            results.append(self.agents_dynamic_status[k])
        return results

    def update_agent_status(self, id='-1', status=None):
        new_value = {id: status}
        self.agents_dynamic_status.update(new_value)

    def get_global_status(self, myid=-1):
        # 获得在场景中，全局坐标系下表示的本体AGV和其他AGV的状态，并按照距离排序。
        self_status = self.agents_dynamic_status.get(myid)
        other_status = self.get_other_agents_status(myid)

        self_global_status = [self_status.px, self_status.py, self_status.vx, self_status.vy, self_status.last_local_gx,
                              self_status.last_local_gy, self_status.next_local_gx, self_status.next_local_gy,
                              self_status.global_gx, self_status.global_gy, self_status.radius, self_status.v_mean]

        other_global_status = list()
        for item in other_status:
            other_dis = math.sqrt((item.px - self_status.px)**2 + (item.py - self_status.py)**2)
            other_global_status.append([other_dis, item.px, item.py, item.vx, item.vy, item.last_local_gx,
                                        item.last_local_gy, item.next_local_gx, item.next_local_gy, item.global_gx,
                                        item.global_gy, item.radius+self_status.radius, item.v_mean])

        if len(other_global_status) >= 2:
            for i in range(1, len(other_global_status)):
                for j in range(0, len(other_global_status)-i):
                    if other_global_status[j][0] < other_global_status[j+1][0]:
                        other_global_status[j], other_global_status[j+1] = other_global_status[j+1], \
                                                                           other_global_status[j]

        if len(self_status.shortest_path_coordinate) >= 5:
            self_shortest_path_coordinate = list(reversed(self_status.shortest_path_coordinate))[-5:]
        else:
            self_shortest_path_coordinate = list(reversed(self_status.shortest_path_coordinate))

        current_global_status = GlobalStatus(self_global_status, other_global_status, self_shortest_path_coordinate, self_status.mission_status)

        return current_global_status, other_status

    def get_local_full_status(self, myid=-1):
        other_status_local = []
        self_status = self.agents_dynamic_status.get(myid)
        other_status = self.get_other_agents_status(myid)
        # for i in range(len(other_status)):
        #     print("other_status", other_status[i].px, other_status[i].py, other_status[i].vx, other_status[i].vy,
        #           other_status[i].gx, other_status[i].gy, other_status[i].dis_road, other_status[i].dis_area,
        #           other_status[i].dis_other, other_status[i].radius)
        # self state
        local_goal_x = (self_status.gx-self_status.px) / 5.0
        if local_goal_x > 1.0:
            local_goal_x = 1.0
        if local_goal_x < -1.0:
            local_goal_x = -1.0

        local_goal_y = (self_status.gy - self_status.py) / 5.0
        if local_goal_y > 1.0:
            local_goal_y = 1.0
        if local_goal_y < -1.0:
            local_goal_y = -1.0

        self_status_local = [self_status.vx, self_status.vy, local_goal_x, local_goal_y, np.sqrt(local_goal_x**2+local_goal_y**2), self_status.radius]

        # self observation
        self_obervation_local = self_status.allow_action

        # other status
        for i in range(len(other_status)):
            other_status_local.append([other_status[i].px-self_status.px, other_status[i].py-self_status.py, other_status[i].vx-self_status.vx,
                                  other_status[i].vy-self_status.vy, other_status[i].gx-self_status.px, other_status[i].gy-self_status.py,
                                  np.sqrt((other_status[i].px-self_status.px)**2 + (other_status[i].py-self_status.py)**2),
                                  other_status[i].radius, other_status[i].radius+self_status.radius])

        if len(other_status_local) >= 2:
            for i in range(1, len(other_status_local)):
                for j in range(0, len(other_status_local)-i):
                    if other_status_local[j][6] < other_status_local[j+1][6]:
                        other_status_local[j], other_status_local[j+1] = other_status_local[j+1], other_status_local[j]

        # local map
        image_position_axis1 = int((self_status.px/self.map_width)*int(self.map_width/GoConfig.MAP_RESOLUTION))
        image_position_axis0 = int((self_status.py/self.map_height)*int(self.map_height/GoConfig.MAP_RESOLUTION))
        # local_map_image_1 = Image.fromarray(self.staic_map_array.astype('uint8'))
        # local_map_image_1.show()
        # local_map_array = self.staic_map_array[0:224, 100:324]
        local_map_array = self.staic_map_array[(int(self.map_height/GoConfig.MAP_RESOLUTION)-image_position_axis0):
                                               ((int(self.map_height/GoConfig.MAP_RESOLUTION)-image_position_axis0)
                                                + GoConfig.HEIGHT), image_position_axis1:(image_position_axis1 + GoConfig.WIDTH)].copy()
        # local_map_image_2 = Image.fromarray(local_map_array.astype('uint8'))
        # draw_local_map_image_1 = ImageDraw.Draw(local_map_image_2)
        # title1 = str(self_status.px)
        # title2 = str(self_status.py)
        # title3 = str(image_position_axis0)
        # title4 = str(image_position_axis1)
        # draw_local_map_image_1.text((0, 0), title1+', '+title2+', '+title3+', '+title4)
        # local_map_image_2.show()
        current_local_full_status = LocalFullStatus(local_map_array, self_obervation_local, self_status_local, other_status_local)
        return current_local_full_status, other_status

    def get_local_full_polar_status(self, myid=-1):
        other_status_local = []
        self_status = self.agents_dynamic_status.get(myid)
        other_status = self.get_other_agents_status(myid)

        # self state
        self_vel = math.sqrt(self_status.vx**2 + self_status.vy**2)
        self_vel_theta = math.atan2(self_status.vy, self_status.vx)

        if - math.pi <= self_vel_theta <= 0:
            self_vel_theta = self_vel_theta + math.pi * 2

        self_last_local_goal = math.sqrt((self_status.last_local_gx - self_status.px) ** 2 + (self_status.last_local_gy - self_status.py) ** 2)
        self_last_local_goal_theta = math.atan2(self_status.last_local_gy - self_status.py, self_status.last_local_gx - self_status.px)
        if - math.pi <= self_last_local_goal_theta <= 0:
            self_last_local_goal_theta = self_last_local_goal_theta + math.pi * 2

        self_next_local_goal = math.sqrt((self_status.next_local_gx - self_status.px)**2 + (self_status.next_local_gy - self_status.py)**2)
        self_next_local_goal_theta = math.atan2(self_status.next_local_gy - self_status.py, self_status.next_local_gx - self_status.px)
        if - math.pi <= self_next_local_goal_theta <= 0:
            self_next_local_goal_theta = self_next_local_goal_theta + math.pi * 2

        self_global_goal = math.sqrt((self_status.global_gx - self_status.px)**2 + (self_status.global_gy - self_status.py)**2)
        self_global_goal_theta = math.atan2(self_status.global_gy - self_status.py, self_status.global_gx - self_status.px)
        if - math.pi <= self_global_goal_theta <= 0:
            self_global_goal_theta = self_global_goal_theta + math.pi * 2

        self_status_local = [self_vel, self_vel_theta, self_last_local_goal, self_last_local_goal_theta,
                             self_next_local_goal, self_next_local_goal_theta,
                             self_global_goal, self_global_goal_theta, self_status.v_mean, self_status.radius]

        # self observation
        self_observation_local = self_status.allow_action

        # other status
        # for i in range(len(other_status)):
        #     other_pos = math.sqrt((other_status[i].px - self_status.px)**2 + (other_status[i].py - self_status.py)**2)
        #     other_pos_theta = math.atan2(other_status[i].py - self_status.py, other_status[i].px - self_status.px)
        #     if - math.pi <= other_pos_theta <= 0:
        #         other_pos_theta = other_pos_theta + math.pi * 2
        #
        #     other_vel = math.sqrt((other_status[i].vx - self_status.vx)**2 + (other_status[i].vy-self_status.vy)**2)
        #     other_vel_theta = math.atan2(other_status[i].vy - self_status.vy, other_status[i].vx - self_status.vx)
        #     if - math.pi <= other_vel_theta <= 0:
        #         other_vel_theta = other_vel_theta + math.pi * 2
        #
        #     other_goal = math.sqrt((other_status[i].gx - self_status.px)**2 + (other_status[i].gy - self_status.py)**2)
        #     other_goal_theta = math.atan2(other_status[i].gy - self_status.py, other_status[i].gx - self_status.px)
        #     if - math.pi <= other_goal_theta <= 0:
        #         other_goal_theta = other_goal_theta + math.pi * 2
        #
        #     other_status_local.append([other_pos, other_pos_theta, other_vel, other_vel_theta, other_goal, other_goal_theta,
        #                                other_status[i].radius, other_status[i].radius+self_status.radius])

        # faster version
        for item in other_status:
            other_pos = math.sqrt((item.px - self_status.px)**2 + (item.py - self_status.py)**2)
            other_pos_theta = math.atan2(item.py - self_status.py, item.px - self_status.px)
            if - math.pi <= other_pos_theta <= 0:
                other_pos_theta = other_pos_theta + math.pi * 2

            other_vel = math.sqrt((item.vx - self_status.vx)**2 + (item.vy - self_status.vy)**2)
            other_vel_theta = math.atan2(item.vy - self_status.vy, item.vx - self_status.vx)
            if - math.pi <= other_vel_theta <= 0:
                other_vel_theta = other_vel_theta + math.pi * 2

            other_last_local_goal = math.sqrt((item.last_local_gx - self_status.px)**2 + (item.last_local_gy - self_status.py)**2)
            other_last_local_goal_theta = math.atan2(item.last_local_gy - self_status.py, item.last_local_gx - self_status.px)
            if - math.pi <= other_last_local_goal_theta <= 0:
                other_last_local_goal_theta = other_last_local_goal_theta + math.pi * 2

            other_next_local_goal = math.sqrt((item.next_local_gx - self_status.px)**2 + (item.next_local_gy - self_status.py)**2)
            other_next_local_goal_theta = math.atan2(item.next_local_gy - self_status.py, item.next_local_gx - self_status.px)
            if - math.pi <= other_next_local_goal_theta <= 0:
                other_next_local_goal_theta = other_next_local_goal_theta + math.pi * 2

            other_global_goal = math.sqrt((item.global_gx - self_status.px)**2 + (item.global_gy - self_status.py)**2)
            other_global_goal_theta = math.atan2(item.global_gy - self_status.py, item.global_gx - self_status.px)
            if - math.pi <= other_global_goal_theta <= 0:
                other_global_goal_theta = other_global_goal_theta + math.pi * 2

            # other_route_length = self.shortest_length_node_to_node(self_status.px, self_status.py, item.px, item.py,
            #                                                        self.G, self.road_node_Nos, self.road_node_info,
            #                                                        self.road_lines, self.road_lines_num,
            #                                                        self.node_edges)

            other_status_local.append([other_pos, other_pos_theta, other_vel, other_vel_theta, other_last_local_goal,
                                       other_last_local_goal_theta, other_next_local_goal, other_next_local_goal_theta,
                                       other_global_goal, other_global_goal_theta, item.radius,
                                       item.radius + self_status.radius])

        if len(other_status_local) >= 2:
            for i in range(1, len(other_status_local)):
                for j in range(0, len(other_status_local)-i):
                    if other_status_local[j][0] < other_status_local[j+1][0]:
                        other_status_local[j], other_status_local[j+1] = other_status_local[j+1], other_status_local[j]

        # if len(other_status_local) < 7:
        #     print("id:1st", self_status.id)
        # length_agent_s_to_agent_e = self.shortest_length_node_to_nodes(self_status.px, self_status.py, other_status,
        #                                                                self.G, self.road_node_Nos, self.road_node_info,
        #                                                                self.road_lines, self.node_edges)

        # for i in range(len(other_status)):
        #     other_status_local[i].insert(6, length_agent_s_to_agent_e[i])
        #
        # if len(other_status_local) >= 2:
        #     # other_status_local_np = np.array(other_status_local)
        #     # other_status_local_np = self.sorted_by_distance(other_status_local_np, 6)
        #     # other_status_local = other_status_local_np.tolist()
        #     for i in range(1, len(other_status_local)):
        #         for j in range(0, len(other_status_local)-i):
        #             if other_status_local[j][6] < other_status_local[j+1][6]:
        #                 other_status_local[j], other_status_local[j+1] = other_status_local[j+1], other_status_local[j]
        #             if other_status_local[j][6] == other_status_local[j+1][6]:
        #                 if other_status_local[j][0] < other_status_local[j+1][0]:
        #                     other_status_local[j], other_status_local[j + 1] = other_status_local[j + 1], other_status_local[j]

        # local map
        # image_position_axis1 = int((self_status.px/self.map_width)*int(self.map_width/GoConfig.MAP_RESOLUTION))
        # image_position_axis0 = int((self_status.py/self.map_height)*int(self.map_height/GoConfig.MAP_RESOLUTION))
        #
        # local_map_array = self.staic_map_array[(int(self.map_height/GoConfig.MAP_RESOLUTION)-image_position_axis0):
        #                                        ((int(self.map_height/GoConfig.MAP_RESOLUTION)-image_position_axis0)
        #                                         + GoConfig.HEIGHT), image_position_axis1:(image_position_axis1 +
        #                                                                                   GoConfig.WIDTH)].copy()

        current_local_full_status = LocalFullStatus(self_observation_local, self_status_local, other_status_local)

        return current_local_full_status, other_status

    def full_status_to_numpy(self, status):
        self_np=[]
        pass

    @staticmethod
    @jit(nopython=True)
    def sorted_by_distance(other_status_local, index):
        for i in np.arange(1, other_status_local.shape[0]):
            for j in np.arange(0, other_status_local.shape[0] - i):
                if other_status_local[j][index] < other_status_local[j + 1][index]:
                    other_status_local[j], other_status_local[j + 1] = other_status_local[j + 1], other_status_local[j]
        return other_status_local

    def init_map(self, map_width, map_height, resolution):
        # 加载地图，处理地图等
        if self.map_pic_str is not None:
            # load image
            map_global = Image.open(self.map_pic_str).convert('RGB')
            map_global = map_global.resize((int(map_width/resolution), int(map_height/resolution)), Image.ANTIALIAS)
            # transform to numpy
            map_img = np.array(map_global)
            # show map
            # local_map_image_1 = Image.fromarray(map_img.astype('uint8'))
            # local_map_image_1.show()
            # add axis 0
            axis_0 = np.zeros((int(GoConfig.HEIGHT/2), int(map_width/resolution), 3), dtype=np.int)
            map_img = np.concatenate((map_img, axis_0), axis=0)
            map_img = np.concatenate((axis_0, map_img), axis=0)
            # add axis 1
            axis_1 = np.zeros((int(map_height/resolution)+GoConfig.HEIGHT, int(GoConfig.WIDTH/2), 3), dtype=np.int)
            map_img = np.concatenate((map_img, axis_1), axis=1)
            map_img = np.concatenate((axis_1, map_img), axis=1)
            # show map
            # local_map_image_1 = Image.fromarray(map_img.astype('uint8'))
            # local_map_image_1.show()
            self.staic_map_array = map_img

    def init_road_network(self, road_config_file):
        """
        with open(road_config_file) as f:
            # read the context in "road_grid" text
            list_text = []
            for line in f.readlines():
                list_text.append(line.strip().split('\n'))
            image_height = int(list_text[0][0])
            road_lines_num = int(list_text[1][0])

            # extract the road lines
            road_lines = np.zeros((road_lines_num, 4))
            row = 0
            for i in range(2, road_lines_num + 2):
                road_line = list_text[i][0].strip().split()
                road_lines[row, :] = road_line[0:4]
                # convert to local world coordinate system
                road_lines[row, 1] = image_height - road_lines[row, 1]
                road_lines[row, 3] = image_height - road_lines[row, 3]
                row += 1

            # extract the coordinates
            road_node_coordinates = road_lines.reshape((road_lines_num * 2, 2))
            road_node_coordinates = np.unique(road_node_coordinates, axis=0)

            # create the node: No. + Coordinate
            road_node_Nos = list(range(1, road_node_coordinates.shape[0] + 1))
            road_node_info = dict(zip(road_node_Nos, road_node_coordinates))

            # create edges
            node_edges = []
            for i in range(road_lines_num):
                start_point = np.array([road_lines[i][0], road_lines[i][1]])
                start_index = None
                end_point = np.array([road_lines[i][2], road_lines[i][3]])
                end_index = None
                for (key, value) in road_node_info.items():
                    if (value == start_point).all() and (not start_index):
                        start_index = key
                    if (value == end_point).all() and (not end_index):
                        end_index = key
                node_edges.append([start_index, end_index, {
                    'len': np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)}])
        return road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges
        """
        # 读取路径文件
        with open(road_config_file) as tx:
            list_text = []
            for line in tx.readlines():
                list_text.append(line.strip().split('\n'))
            roads_num = int(list_text[0][0])
            stations_num = int(list_text[int(list_text[0][0]) + 1][0])
            roads_text = list_text[1: roads_num + 1]
            stations_text = list_text[roads_num + 2: roads_num + stations_num + 2]

        # 提取路径的起止节点坐标
        road_lines = np.zeros((roads_num, 4))
        road_directions = np.zeros(roads_num)
        for i in roads_text:
            road_line = i[0].strip().split()
            road_lines[roads_text.index(i), :] = road_line[0:4]
            road_directions[roads_text.index(i)] = road_line[4]

        # 提取站点的名称和坐标
        station_codes = np.zeros(stations_num, dtype=np.int32)
        station_coordinates = np.zeros((stations_num, 2))
        for i in stations_text:
            station = i[0].strip().split()
            station_codes[stations_text.index(i)] = station[0]
            station_coordinates[stations_text.index(i)] = station[1:3]
        stationInfo = dict(zip(station_codes, station_coordinates))

        # 生成边
        node_edges = []
        for i in range(roads_num):
            start_point = np.array([road_lines[i][0], road_lines[i][1]])
            start_index = None
            end_point = np.array([road_lines[i][2], road_lines[i][3]])
            end_index = None
            for (key, value) in stationInfo.items():
                if (value == start_point).all() and (not start_index):
                    start_index = key
                if (value == end_point).all() and (not end_index):
                    end_index = key
            road_direction = road_directions[i]
            if road_direction == 0:
                node_edges.append([start_index, end_index, {
                    'len': np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)}])
            elif road_direction == 1:
                node_edges.append([end_index, start_index, {
                    'len': np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)}])
            elif road_direction == 2:
                node_edges.append([start_index, end_index, {
                    'len': np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)}])
                node_edges.append([end_index, start_index, {
                    'len': np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)}])
            else:
                raise ValueError('wrong direction')

        road_node_Nos = station_codes
        road_node_info = stationInfo
        road_lines_num = roads_num

        return road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges


    def shortest_length_node_to_node(self, node_s_x, node_s_y, node_e_x, node_e_y, G, road_node_Nos, road_node_info,
                                    road_lines, road_lines_num, node_edges):
        """
        calculate the shortest length from the start node 's' to the end node 'e'
        :param node_s_x:
        :param node_s_y:
        :param node_e_x:
        :param node_e_y:
        :param G:
        :param road_node_Nos:
        :param road_node_info:
        :param road_lines:
        :param road_lines_num:
        :param node_edges:
        :return:
        """
        # start = time.time()
        # the start node, s
        agent_node_No = []
        agent_node_No.append(0)
        agent_node_coordinate = np.zeros((2, 2))
        agent_node_coordinate[0][0] = int(node_s_x / 0.05)
        agent_node_coordinate[0][1] = int(node_s_y / 0.05)

        # the end node, e
        agent_node_No.append(len(road_node_Nos) + 1)
        agent_node_coordinate[1][0] = int(node_e_x / 0.05)
        agent_node_coordinate[1][1] = int(node_e_y / 0.05)
        road_node_info.update(dict(zip(agent_node_No, agent_node_coordinate)))

        # add node
        env_node_Nos = agent_node_No + road_node_Nos
        G.add_nodes_from(env_node_Nos)

        # start = time.time()
        # add edges from agent to the nearest road line
        agent_line_shortest_node = np.zeros((2, 2))
        for k in range(2):
            # calculate the distance from the agent to the lines
            # agent_line_dist = []
            # for i in range(road_lines_num):
            #     dist = self.distance_p2seg(road_lines[i][0], road_lines[i][1], road_lines[i][2], road_lines[i][3],
            #                                agent_node_coordinate[k][0], agent_node_coordinate[k][1])
            #     agent_line_dist.append(dist)
            agent_line_shortest_index = self.distance_p2network(road_lines, agent_node_coordinate[k][0],
                                                                agent_node_coordinate[k][1])

            # find the shortest line's node
            for (key, value) in road_node_info.items():
                if value[0] == road_lines[agent_line_shortest_index][0] and value[1] == \
                        road_lines[agent_line_shortest_index][1]:
                    agent_line_shortest_node[k][0] = key
                if value[0] == road_lines[agent_line_shortest_index][2] and value[1] == \
                        road_lines[agent_line_shortest_index][3]:
                    agent_line_shortest_node[k][1] = key

            # add new edges from the agent node to road note
            node_edges.append([agent_node_No[k], agent_line_shortest_node[k][0], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][0]][0] - agent_node_coordinate[k][0]) ** 2 + (
                            road_node_info[agent_line_shortest_node[k][0]][1] - agent_node_coordinate[k][1]) ** 2)}])
            node_edges.append([agent_node_No[k], agent_line_shortest_node[k][1], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][1]][0] - agent_node_coordinate[k][0]) ** 2 + (
                            road_node_info[agent_line_shortest_node[k][1]][1] - agent_node_coordinate[k][1]) ** 2)}])
        # end = time.time()
        # print("~~~~~~~~~~~~~~~~~~~~~~~shortest_length_node_to_node time~~~~~~~~~~~~~~~~~~`", start-end)

        G.add_edges_from(node_edges)

        length_agent_s_to_agent_e = nx.dijkstra_path_length(G, source=agent_node_No[0], target=agent_node_No[1], weight='len')


        for k in range(2):
            node_edges.remove([agent_node_No[k], agent_line_shortest_node[k][0], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][0]][0] - agent_node_coordinate[k][0]) ** 2 + (
                            road_node_info[agent_line_shortest_node[k][0]][1] - agent_node_coordinate[k][1]) ** 2)}])
            node_edges.remove([agent_node_No[k], agent_line_shortest_node[k][1], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][1]][0] - agent_node_coordinate[k][0]) ** 2 + (
                            road_node_info[agent_line_shortest_node[k][1]][1] - agent_node_coordinate[k][1]) ** 2)}])

        G.clear()
        return length_agent_s_to_agent_e * 0.05

    @staticmethod
    @nb.vectorize(nopython=True)
    def distance_p2seg(seg_0x, seg_0y, seg_1x, seg_1y, p_x, p_y):
        cross = (seg_1x - seg_0x) * (p_x - seg_0x) + (seg_1y - seg_0y) * (p_y - seg_0y)

        if cross <= 0:
            return np.sqrt((p_x - seg_0x) ** 2 + (p_y - seg_0y) ** 2)

        d2 = (seg_1x - seg_0x) ** 2 + (seg_1y - seg_0y) ** 2

        if cross >= d2:
            return np.sqrt((p_x - seg_1x) ** 2 + (p_y - seg_1y) ** 2)

        r = cross / d2
        p0 = seg_0x + (seg_1x - seg_0x) * r
        p1 = seg_0y + (seg_1y - seg_0y) * r
        return np.sqrt((p_x - p0) ** 2 + (p_y - p1) ** 2)

    @staticmethod
    @jit(nopython=True, nogil=True, parallel=True)
    def distance_p2network(road_lines, p_x, p_y):
        agent_line_dist = np.zeros(road_lines.shape[0])
        for i in np.arange(road_lines.shape[0]):
            cross = (road_lines[i][2] - road_lines[i][0]) * (p_x - road_lines[i][0]) \
                    + (road_lines[i][3] - road_lines[i][1]) * (p_y - road_lines[i][1])

            if cross <= 0:
                agent_line_dist[i] = np.sqrt((p_x - road_lines[i][0]) ** 2 + (p_y - road_lines[i][1]) ** 2)
                continue

            d2 = (road_lines[i][2] - road_lines[i][0]) ** 2 + (road_lines[i][3] - road_lines[i][1]) ** 2
            if cross >= d2:
                agent_line_dist[i] = np.sqrt((p_x - road_lines[i][2]) ** 2 + (p_y - road_lines[i][3]) ** 2)
                continue

            r = cross / d2
            p0 = road_lines[i][0] + (road_lines[i][2] - road_lines[i][0]) * r
            p1 = road_lines[i][1] + (road_lines[i][3] - road_lines[i][1]) * r
            agent_line_dist[i] = np.sqrt((p_x - p0) ** 2 + (p_y - p1) ** 2)

        # find the nearest line index
        agent_line_dist_shortest = 99999
        agent_line_shortest_index = 0
        for i in np.arange(agent_line_dist.shape[0]):
            if agent_line_dist[i] < agent_line_dist_shortest:
                agent_line_shortest_index = i
                agent_line_dist_shortest = agent_line_dist[i]

        return agent_line_shortest_index

    def shortest_length_node_to_nodes(self, node_s_x, node_s_y, nodes_e, G, road_node_Nos, road_node_info,
                                    road_lines, node_edges):
        """
        calculate the shortest length from the start node 's' to the list of end nodes 'e'
        :param node_s_x:
        :param node_s_y:
        :param node_e_x:
        :param node_e_y:
        :param G:
        :param road_node_Nos:
        :param road_node_info:
        :param road_lines:
        :param road_lines_num:
        :param node_edges:
        :return:
        """
        agent_node_No = []
        agent_node_coordinate = np.zeros((len(nodes_e)+1, 2))
        # the start node, s
        agent_node_No.append(0)
        agent_node_coordinate[0][0] = int(node_s_x / 0.05)
        agent_node_coordinate[0][1] = int(node_s_y / 0.05)

        # the end node, e
        for i in range(len(nodes_e)):
            agent_node_No.append(len(road_node_Nos) + 1 + i)
            agent_node_coordinate[i + 1][0] = int(nodes_e[i].px / 0.05)
            agent_node_coordinate[i + 1][1] = int(nodes_e[i].py / 0.05)

        # road_node_info.update(dict(zip(agent_node_No, agent_node_coordinate)))

        # add node
        env_node_Nos = agent_node_No + road_node_Nos

        # add edges from agent to the nearest road line
        agent_line_shortest_node = np.zeros((len(nodes_e)+1, 2))
        for k in range(len(nodes_e)+1):
            agent_line_shortest_index = self.distance_p2network(road_lines, agent_node_coordinate[k][0],
                                                                agent_node_coordinate[k][1])

            # find the shortest line's node
            for (key, value) in road_node_info.items():
                if value[0] == road_lines[agent_line_shortest_index][0] and value[1] == \
                        road_lines[agent_line_shortest_index][1]:
                    agent_line_shortest_node[k][0] = key
                if value[0] == road_lines[agent_line_shortest_index][2] and value[1] == \
                        road_lines[agent_line_shortest_index][3]:
                    agent_line_shortest_node[k][1] = key
                if agent_line_shortest_node[k][0] != 0 and agent_line_shortest_node[k][1] != 0:
                    break

        # calculate the distance from the start agent to each agent, and return a list
        length_agent_s_to_agent_e = []
        for k in range(1, len(nodes_e) + 1):
            # add edges from the start agent node to the nearest road
            node_edges.append([agent_node_No[0], agent_line_shortest_node[0][0], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[0][0]][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node[0][0]][1] - agent_node_coordinate[0][1]) ** 2)}])
            node_edges.append([agent_node_No[0], agent_line_shortest_node[0][1], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[0][1]][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node[0][1]][1] - agent_node_coordinate[0][1]) ** 2)}])

            # add edges from the k th agent node to the nearest road
            node_edges.append([agent_node_No[k], agent_line_shortest_node[k][0], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][0]][0] - agent_node_coordinate[k][0]) ** 2 + (
                            road_node_info[agent_line_shortest_node[k][0]][1] - agent_node_coordinate[k][1]) ** 2)}])
            node_edges.append([agent_node_No[k], agent_line_shortest_node[k][1], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][1]][0] - agent_node_coordinate[k][0]) ** 2 + (
                            road_node_info[agent_line_shortest_node[k][1]][1] - agent_node_coordinate[k][1]) ** 2)}])

            G.add_nodes_from(env_node_Nos)
            G.add_edges_from(node_edges)

            length_agent_s_to_agent_e.append(0.05 * nx.dijkstra_path_length(G, source=agent_node_No[0],
                                                                            target=agent_node_No[k], weight='len'))

            # remove edges from the start agent node to the nearest road
            node_edges.remove([agent_node_No[0], agent_line_shortest_node[0][0], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[0][0]][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node[0][0]][1] - agent_node_coordinate[0][1]) ** 2)}])
            node_edges.remove([agent_node_No[0], agent_line_shortest_node[0][1], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[0][1]][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node[0][1]][1] - agent_node_coordinate[0][1]) ** 2)}])

            # remove edges from the k th agent node to the nearest road
            node_edges.remove([agent_node_No[k], agent_line_shortest_node[k][0], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][0]][0] - agent_node_coordinate[k][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node[k][0]][1] - agent_node_coordinate[k][1]) ** 2)}])
            node_edges.remove([agent_node_No[k], agent_line_shortest_node[k][1], {'len': np.sqrt(
                (road_node_info[agent_line_shortest_node[k][1]][0] - agent_node_coordinate[k][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node[k][1]][1] - agent_node_coordinate[k][1]) ** 2)}])

            G.clear()
        return length_agent_s_to_agent_e
