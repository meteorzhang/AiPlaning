import sys
sys.path.append('../')
from GoFitnetss import GoFitness
from pysim2d import pysim2d
from GoConfig import GoConfig
from GoStatus import AgentState
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import math
import time
from numba import jit
import copy


class GoEmulator:
    def __init__(self, id=-1, share_env=None, enable_show=False):
        self.id = id
        self.emulator = None
        self.share_env = share_env
        self.fitness = GoFitness()
        self._start_x = 0
        self._start_y = 0
        self._start_orientation = 0.0
        self.virtual_laser = []
        self.radius = GoConfig.RADIUS
        self.nearest_road_dis = 0
        self.nearest_allow_area_dis = 0
        self.nearest_other_agent_dis = 0
        self.dis_agent_to_goal = 0
        self.G = nx.DiGraph()
        self.shortest_path_nodes = []
        self.shortest_path_nodes_pointer = 1
        self._global_goal_px = 0
        self._global_goal_py = 0
        self._global_goal_orientation = 0.0
        self._last_local_goal_px = 0
        self._last_local_goal_py = 0
        self._next_local_goal_px = 0
        self._next_local_goal_py = 0
        self.node_0_x = 0
        self.node_0_y = 0
        self.observation_angle = GoConfig.OBSERVATION_ANGLE
        self.v_mean = np.array([0.5, 0.5, 0.5])
        self.v_last = 0
        self.simple_paths = list()
        self.has_path = False
        self.allow_action = list()
        self.next_station = list()
        self._robot_vx = 0
        self._robot_vy = 0
        self._robot_px = 0
        self._robot_py = 0
        self._env_done = 0
        self._deadlock = False
        self.shortest_path_coordinate = list()
        self.selected_path = None

        if self.share_env is not None:
            # self.map_url = share_env.map_pic_str
            # self.one_channel_map_pic_str = share_env.one_channel_map_pic_str
            self.road_url = share_env.roads_gunplot_cfg_url
            self.target_range = float(share_env.target_range)
            # self.allow_area_url = share_env.allow_area_gunplot_cfg_url
            self.enable_show = enable_show
            self.emulator = pysim2d.pysim2d()
            # self.emulator.init(self.one_channel_map_pic_str, self.allow_area_url, self.road_url,
            #                    self.enable_show, False, False)
            self.emulator.init_road_map(self.road_url, self.enable_show)
            self.emulator.set_parameter(self.radius, self.radius, np.pi, 0, GoConfig.OBSERVATION_ANGLE,
                                        GoConfig.OBSERVATION_RANGE, self.target_range)
            # title = "Map:%s Agent ID:%d" % (self.share_env.name, int(self.id))
            self.emulator.set_title(int(self.share_env.name), int(self.id))

        self.road_node_Nos, self.road_node_info, self.road_lines, self.road_directions, self.road_lines_num, \
            self.node_edges = self._init_road_network(self.road_url)

        self.node_labels = dict(zip(self.road_node_Nos, self.road_node_Nos))

        # self.G.add_nodes_from(self.road_node_Nos)
        # self.G.add_edges_from(self.node_edges)
        # nx.draw_networkx_nodes(self.G, self.road_node_info, node_size=50, node_color="#6CB6FF")  # node
        # nx.draw_networkx_edges(self.G, self.road_node_info, self.node_edges)  # edge
        # nx.draw_networkx_labels(self.G, self.road_node_info, self.node_labels)  # label
        # plt.show()

        # self.emulator.visualize("road.dat", "jinxing.dat")
        # def visualize(self,enable=False):
        # self.emulator.visilize()
        #    pass

    def _init_road_network(self, road_config_file):
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

        return road_node_Nos, road_node_info, road_lines, road_directions, road_lines_num, node_edges


    def convert_other_status_to_gnuplot(self, o_status):
        others = []
        if o_status is not None:
            for idx in range(len(o_status)):
                state = o_status[idx]
                others.append(state.id)
                others.append(state.px)
                others.append(state.py)
                others.append(state.radius)
        return others

    def agents_average_velocity(self, v, other_agents):
        agents_velocity = np.zeros(len(other_agents)+1)
        agents_velocity[0] = v
        for i in range(len(other_agents)):
            state = other_agents[i]
            agents_velocity[i+1] = np.sqrt(state.vx**2 + state.vy**2)
        return np.mean(agents_velocity)

    def block(self, v, nearest_other_agent_dis):
        goal_dis = math.sqrt((self._next_local_goal_px-self._robot_px)**2+(self._next_local_goal_py-self._robot_py)**2)
        if GoConfig.STEPPING_VELOCITY * GoConfig.STEPPING_TIME < goal_dis:
            step_dis = GoConfig.STEPPING_VELOCITY * GoConfig.STEPPING_TIME
        else:
            step_dis = goal_dis
        if 2 * GoConfig.RADIUS <= nearest_other_agent_dis <= 2 * GoConfig.RADIUS + step_dis:
            # velocity mean
            for i in range(1, self.v_mean.shape[0]):
                self.v_mean[i - 1] = self.v_mean[i]
            self.v_mean[-1] = v
            v_mean = np.mean(self.v_mean)

            if v_mean == 0 and v == 0:
                return True
            else:
                return False
        else:
            return False

    def step(self, v: float, other_agent_array: list, step_time: float = 1.0):
        """
        在模拟环境中执行1步动作
        :param v:
        :param other_agent_array:
        :param step_time:
        :return:
        """
        # 若执行动作将发生终点死锁，则选择静止
        forced_stop = 0
        if len(other_agent_array) > 0 and v > 0.1:
            # a_v = self._actions.get_action_by_indx(action)
            # 到下一个站点的距离
            d_next = math.sqrt((self._robot_px - self._next_local_goal_px) ** 2 + (self._robot_py - self._next_local_goal_py) ** 2)
            # 对下一个站点的死锁判断
            if d_next < 3 * v and len(self.shortest_path_coordinate) > 2:
                self_simple_paths = copy.deepcopy(self.simple_paths)
                for path in self_simple_paths:
                    path.remove(path[0])
            else:
                self_simple_paths = copy.deepcopy(self.simple_paths)

            if self.check_terminal_deadlock(self_simple_paths, other_agent_array):
                v = 0
                forced_stop = 1

            if d_next < 3 * v and v != 0 and len(self.shortest_path_coordinate) > 2:
                for path in self_simple_paths:
                    path.remove(path[0])

                if self.check_terminal_deadlock(self_simple_paths, other_agent_array):
                    v = 0
                    forced_stop = 1

        # 转换其他智能体状态，用于GNU显示
        gnu_others = self.convert_other_status_to_gnuplot(other_agent_array)
        # 计算速度
        length = math.sqrt((self._next_local_goal_px-self._robot_px)**2+(self._next_local_goal_py-self._robot_py)**2)
        if length == 0:
            self._robot_vx = 0
            self._robot_vy = 0
        else:
            self._robot_vx = round(v*(self._next_local_goal_px-self._robot_px)/length, 3)
            self._robot_vy = round(v*(self._next_local_goal_py-self._robot_py)/length, 3)
        # 模拟器执行1步
        self.emulator.step_road_map(self._robot_vx, self._robot_vy, self._next_local_goal_px, self._next_local_goal_py,
                                    step_time, gnu_others)
        # self.emulator.visualize()
        # Get status after taking action
        self._robot_px = round(self.emulator.get_robot_pose_x(), 3)
        self._robot_py = round(self.emulator.get_robot_pose_y(), 3)
        # env_robot_orientation = self.emulator.get_robot_orientation()
        self._env_done = self.emulator.done()
        # nearest_road_dis = self.emulator.get_road_dis()
        # nearest_allow_area_dis = self.emulator.get_tongxing_dis()
        nearest_other_agent_dis = self.emulator.get_other_robots_dis()
        self._global_goal_px = round(self.emulator.get_goal_pose_x(), 3)
        self._global_goal_py = round(self.emulator.get_goal_pose_y(), 3)
        # agents_average_velocity = self.agents_average_velocity(v, other_agent_array)
        # env_goal_orientation = self.emulator.get_goal_orientation()
        # self.virtual_laser = [i*50 for i in self.emulator.get_lidar_dis()]
        # self.virtual_laser = self.emulator.get_lidar_dis()
        # virtual_laser = self.emulator.get_lidar_dis()
        # if len(virtual_laser) != GoConfig.OBSERVATION_ANGLE_SIZE:
        #     raise Exception("wrong laser number", len(virtual_laser))

        # velocity mean
        if v > 0:
            for i in range(1, self.v_mean.shape[0]):
                self.v_mean[i-1] = self.v_mean[i]
            self.v_mean[-1] = v

        # ？？？？？？？？？？？？？？？
        # block = self.block(v, nearest_other_agent_dis)

        # self.next_station, self.allow_action = self.search_allow_action(self._robot_px, self._robot_py,
        #                                                                 self.road_node_info, self.simple_paths)

        # 若到达某局部站点，则记录该站点，并简化路径
        if self._env_done == 5 or self._env_done == 3:
            self._last_local_goal_px = self._next_local_goal_px
            self._last_local_goal_py = self._next_local_goal_py

            # 简化路径
            for (key, value) in self.road_node_info.items():
                if value[0] == self._last_local_goal_px and value[1] == self._last_local_goal_py:
                    last_station = key
                    self.simple_paths = self.simplify_paths(self.simple_paths, last_station)
                    break

            # 选择不会发生死锁的最短路径
            self.selected_path = self.select_optimal_path(self.simple_paths, other_agent_array)

            # 更新站点坐标
            self.shortest_path_coordinate.clear()
            for station in self.simple_paths[self.selected_path]:
                self.shortest_path_coordinate.append([self.road_node_info[station][0], self.road_node_info[station][1]])

        # # 判断路径死锁
        # if self._env_done == 3:
        #     self._deadlock = False
        # else:
        #     self._deadlock = self.check_path_deadlock(self.simple_paths, other_agent_array)

        # 判断碰撞死锁
        if not self._deadlock and self._env_done != 3:
            self._deadlock = self.check_collision_deadlock(self._robot_px, self._robot_py, v, self._last_local_goal_px,
                                                           self._last_local_goal_py, self._next_local_goal_px,
                                                           self._next_local_goal_py, other_agent_array)

        # 计算奖励
        reward, done = self.fitness.calculate_road_map_reward(self._env_done, nearest_other_agent_dis,
                                                              self._deadlock, v)

        # 到达最终站点时，刷新站点，并生成路径以及坐标
        if self._env_done == 3:
            self.has_path = False
            while not self.has_path:
                self.emulator.reset_goal()
                self._global_goal_px = round(self.emulator.get_goal_pose_x(), 3)
                self._global_goal_py = round(self.emulator.get_goal_pose_y(), 3)
                self.simple_paths, self.has_path = self.search_paths_agent_to_goal(self._robot_px, self._robot_py,
                                                                                   self._global_goal_px,
                                                                                   self._global_goal_py, self.G,
                                                                                   self.road_node_Nos,
                                                                                   self.road_node_info, self.road_lines,
                                                                                   self.road_directions,
                                                                                   self.road_lines_num,
                                                                                   self.node_edges)

                self.shortest_path_coordinate.clear()
                for station in self.simple_paths[0]:
                    self.shortest_path_coordinate.append([self.road_node_info[station][0],
                                                          self.road_node_info[station][1]])
        # 到达中间站点或者最终站点时，切换局部站点到下一站点
        if self._env_done == 3 or self._env_done == 5:
            self.next_station = self.search_next_station(self._last_local_goal_px, self._last_local_goal_py,
                                                         self.road_node_info, self.simple_paths[self.selected_path])
            self._next_local_goal_px = self.next_station[0]
            self._next_local_goal_py = self.next_station[1]

        # 创建一个新的状态
        status = AgentState(self._robot_px, self._robot_py, self._robot_vx, self._robot_vy, self._last_local_goal_px,
                            self._last_local_goal_py, self._next_local_goal_px, self._next_local_goal_py,
                            self._global_goal_px, self._global_goal_py, self.radius,
                            np.mean(self.v_mean), self.simple_paths, self.shortest_path_coordinate, self._env_done,
                            self.id)

        return status, reward, done, forced_stop

    def reset(self):
        # 生成目标，并确保AGV到目标之间存在路径，提取不多于3条到达目标的路径
        self.has_path = False
        while not self.has_path:
            self.emulator.reset_road_map()
            self._robot_px = round(self.emulator.get_robot_pose_x(), 3)
            self._robot_py = round(self.emulator.get_robot_pose_y(), 3)
            self._robot_vx = 0
            self._robot_vy = 0
            self._global_goal_px = round(self.emulator.get_goal_pose_x(), 3)
            self._global_goal_py = round(self.emulator.get_goal_pose_y(), 3)
            self.simple_paths, self.has_path = self.search_paths_agent_to_goal(self._robot_px, self._robot_py,
                                                                               self._global_goal_px,
                                                                               self._global_goal_py, self.G,
                                                                               self.road_node_Nos,
                                                                               self.road_node_info, self.road_lines,
                                                                               self.road_directions,
                                                                               self.road_lines_num, self.node_edges)

        if len(self.simple_paths) == 0:
            print("no path")

        # 提取路径上节点的坐标
        self.shortest_path_coordinate.clear()
        self.shortest_path_coordinate.append([self._robot_px, self._robot_py])

        self.v_mean = np.array([0.5, 0.5, 0.5])
        self._last_local_goal_px = self._robot_px
        self._last_local_goal_py = self._robot_py
        self._next_local_goal_px = self._robot_px
        self._next_local_goal_py = self._robot_py
        self._env_done = 0
        self._deadlock = False

        status = AgentState(self._robot_px, self._robot_py, self._robot_vx, self._robot_vy, self._last_local_goal_px,
                            self._last_local_goal_py, self._next_local_goal_px, self._next_local_goal_py,
                            self._global_goal_px, self._global_goal_py, self.radius, np.mean(self.v_mean),
                            self.simple_paths, self.shortest_path_coordinate, self._env_done, self.id)

        return status

    def search_paths_agent_to_goal(self, robot_x, robot_y, goal_x, goal_y, G, road_node_Nos, road_node_info,
                                   road_lines, road_directions, road_lines_num, node_edges):
        """
        Generate all simple paths in the graph G from source to target, starting from shortest ones.
        Return True if G has a path from agent to goal.
        """
        # add target node
        target_node_coordinate = np.zeros((1, 2))
        target_node_coordinate[0][0] = goal_x
        target_node_coordinate[0][1] = goal_y
        target_node = None

        for (key, value) in road_node_info.items():
            if math.sqrt((value[0]-target_node_coordinate[0][0])**2 + (value[1]-target_node_coordinate[0][1])**2) <= 0.01:
                target_node = key

        if target_node == 0:
            print(target_node)
            raise Exception("wrong target node", target_node)

        # Check whether the robot is on the road node or not
        at_node = False
        for (key, value) in road_node_info.items():
            if key == 0:
                continue
            if value[0] == robot_x and value[1] == robot_y:
                at_node = True
                agent_node_No = key

        if at_node == False:
            # add agent node
            agent_node_No = 0
            agent_node_coordinate = np.zeros((1, 2))
            agent_node_coordinate[0][0] = robot_x
            agent_node_coordinate[0][1] = robot_y
            agent_node = dict(zip([agent_node_No], agent_node_coordinate))
            road_node_info.update(agent_node)

            # add node
            env_node_Nos = [agent_node_No] + road_node_Nos
            G.add_nodes_from(env_node_Nos)

            # add edges from agent to the nearest road line
            # calculate the distance from the agent to the lines
            agent_line_dist = []
            for i in range(road_lines_num):
                cross = (road_lines[i][2] - road_lines[i][0]) * (agent_node_coordinate[0][0] - road_lines[i][0]) \
                        + (road_lines[i][3] - road_lines[i][1]) * (agent_node_coordinate[0][1] - road_lines[i][1])
                if cross <= 0:
                    agent_line_dist.append(np.sqrt((agent_node_coordinate[0][0] - road_lines[i][0]) ** 2
                                                   + (agent_node_coordinate[0][1] - road_lines[i][1]) ** 2))
                    continue

                d2 = (road_lines[i][2] - road_lines[i][0]) ** 2 + (road_lines[i][3] - road_lines[i][1]) ** 2
                if cross >= d2:
                    agent_line_dist.append(np.sqrt((agent_node_coordinate[0][0] - road_lines[i][2]) ** 2
                                                   + (agent_node_coordinate[0][1] - road_lines[i][3]) ** 2))
                    continue
                r = cross / d2
                p0 = road_lines[i][0] + (road_lines[i][2] - road_lines[i][0]) * r
                p1 = road_lines[i][1] + (road_lines[i][3] - road_lines[i][1]) * r
                agent_line_dist.append(
                    np.sqrt((agent_node_coordinate[0][0] - p0) ** 2 + (agent_node_coordinate[0][1] - p1) ** 2))

            # find the nearest line index
            agent_line_dist_shortest = float("inf")
            agent_line_shortest_index = 0

            for index, item in enumerate(agent_line_dist):
                if item < agent_line_dist_shortest:
                    agent_line_shortest_index = index
                    agent_line_dist_shortest = item

            # find the shortest line's node
            agent_line_shortest_node0 = None
            agent_line_shortest_node1 = None

            for (key, value) in road_node_info.items():
                if value[0] == road_lines[agent_line_shortest_index][0] and value[1] == \
                        road_lines[agent_line_shortest_index][1]:
                    agent_line_shortest_node0 = key
                if value[0] == road_lines[agent_line_shortest_index][2] and value[1] == \
                        road_lines[agent_line_shortest_index][3]:
                    agent_line_shortest_node1 = key

            # add new edges from the agent node to road note
            if road_directions[agent_line_shortest_index] == 0:
                node_edges.append([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
                    (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                                road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])
            elif road_directions[agent_line_shortest_index] == 1:
                node_edges.append([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
                    (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
                                road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
            elif road_directions[agent_line_shortest_index] == 2:
                node_edges.append([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
                    (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
                                road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
                node_edges.append([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
                    (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                                road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])
            else:
                raise ValueError('wrong direction')

            G.add_edges_from(node_edges)
            simple_paths_list = list()
            if agent_node_No not in G or target_node not in G:
                has_path = False
                G.clear()
            else:
                if nx.has_path(G, source=agent_node_No, target=target_node):
                    simple_paths = nx.shortest_simple_paths(G, source=agent_node_No, target=target_node, weight='len')

                    for path in simple_paths:
                        simple_paths_list.append(path)

                    for path in simple_paths_list:
                        if path[1] == agent_line_shortest_node1:
                            path[0] = agent_line_shortest_node0
                        elif path[1] == agent_line_shortest_node0:
                            path[0] = agent_line_shortest_node1
                        else:
                            raise ValueError('First node Error!')

                    remove_paths_list = list()
                    for path in simple_paths_list:
                        for path_rest in simple_paths_list[simple_paths_list.index(path) + 1:]:
                            if path == path_rest[- len(path):]:
                                remove_paths_list.append(path_rest)

                    for remove_path in remove_paths_list:
                        if remove_path in simple_paths_list:
                            simple_paths_list.remove(remove_path)

                    # Choose 1 simple paths
                    if len(simple_paths_list) > 1:
                        simple_paths_list = simple_paths_list[0:1]

                    # remove edges from the agent node to road note
                    if road_directions[agent_line_shortest_index] == 0:
                        node_edges.remove([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])
                    elif road_directions[agent_line_shortest_index] == 1:
                        node_edges.remove([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
                    elif road_directions[agent_line_shortest_index] == 2:
                        node_edges.remove([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
                        node_edges.remove([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])
                    else:
                        raise ValueError('wrong direction')

                    has_path = True
                    G.clear()
                else:
                    # remove edges from the agent node to road note
                    if road_directions[agent_line_shortest_index] == 0:
                        node_edges.remove([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])
                    elif road_directions[agent_line_shortest_index] == 1:
                        node_edges.remove([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
                    elif road_directions[agent_line_shortest_index] == 2:
                        node_edges.remove([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
                        node_edges.remove([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
                            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])
                    else:
                        raise ValueError('wrong direction')

                    has_path = False
                    G.clear()
        else:
            G.add_edges_from(node_edges)
            simple_paths_list = list()
            # 判断站点是否在路网上
            if agent_node_No not in G or target_node not in G:
                has_path = False
                G.clear()
            else:
                # 判断站点和目标间是否存在路径
                if nx.has_path(G, source=agent_node_No, target=target_node):
                    # 提取所有简单路径
                    simple_paths = nx.shortest_simple_paths(G, source=agent_node_No, target=target_node, weight='len')

                    for path in simple_paths:
                        simple_paths_list.append(path)

                    # 移除带有回环的路网
                    remove_paths_list = list()
                    for path in simple_paths_list:
                        for path_rest in simple_paths_list[simple_paths_list.index(path) + 1:]:
                            if path == path_rest[- len(path):]:
                                remove_paths_list.append(path_rest)

                    for remove_path in remove_paths_list:
                        if remove_path in simple_paths_list:
                            simple_paths_list.remove(remove_path)

                    # 提取最多2条路径
                    if len(simple_paths_list) > 2:
                        simple_paths_list = simple_paths_list[0:2]

                    # 确认存在路径
                    has_path = True
                    G.clear()
                else:
                    # 不存在路径
                    has_path = False
                    G.clear()

        return simple_paths_list, has_path

    # @staticmethod
    def search_allow_action(self, robot_x, robot_y, last_local_goal_px, last_local_goal_py, next_local_goal_px, next_local_goal_py, road_node_info, simple_paths):
        """
        search the allow actions according to current position of agent and the paths from the agent to the goal
        """
        # 初始化
        # allow_action = [1., 0, 0, 0, 0, 0, 0, 0, 0]
        # next_station = [[], [], [], [], [], [], [], [], []]
        allow_action = [1., 0.1, 0.1, 0.1, 0.1]
        next_station = [[], [], [], [], []]
        allow_direction = list()

        # AGV位于站点时
        if math.sqrt((last_local_goal_px-robot_x)**2+(last_local_goal_py-robot_y)**2) < 0.01:
            # 查找重合站点的编号
            coincide_node = None
            for path in simple_paths:
                for station in path:
                    if math.sqrt((road_node_info[station][0]-last_local_goal_px)**2 + (road_node_info[station][1]-last_local_goal_py)**2) < 0.01:
                        coincide_node = station
                        break
            # 在简单路径中，查找站点的编号，并计算允许的运动方向
            for path in simple_paths:
                for i in range(len(path)-1):
                    if math.sqrt((road_node_info[coincide_node][0] - road_node_info[path[i]][0]) ** 2
                                 + (road_node_info[coincide_node][1] - road_node_info[path[i]][1]) ** 2) < 0.01:
                        allow_direction.append([road_node_info[path[i+1]][0], road_node_info[path[i+1]][1],
                                                math.atan2((road_node_info[path[i+1]][1]-road_node_info[path[i]][1]),
                                                           (road_node_info[path[i+1]][0]-road_node_info[path[i]][0]))])
        else:
            # last 和 next 局部站点不重合，则直接计算许可方向
            allow_direction.append([next_local_goal_px, next_local_goal_py,
                                    math.atan2((next_local_goal_py - last_local_goal_py), (next_local_goal_px - last_local_goal_px))])

        # elif math.sqrt((next_local_goal_px-robot_x)**2+(next_local_goal_py-robot_y)**2) < 0.01:
        #     # 查找重合站点的编号
        #     coincide_node = None
        #     for path in simple_paths:
        #         for station in path:
        #             if math.sqrt((road_node_info[station][0]-next_local_goal_px)**2 + (road_node_info[station][1]-next_local_goal_py)**2) < 0.01:
        #                 coincide_node = station
        #                 break
        #     # 在简单路径中，查找站点的编号，并计算允许的运动方向
        #     for path in simple_paths:
        #         for i in range(len(path)-1):
        #             if math.sqrt((road_node_info[coincide_node][0] - road_node_info[path[i]][0]) ** 2
        #                          + (road_node_info[coincide_node][1] - road_node_info[path[i]][1]) ** 2) < 0.01:
        #                 allow_direction.append([road_node_info[path[i+1]][0], road_node_info[path[i+1]][1],
        #                                         math.atan2((road_node_info[path[i+1]][1]-road_node_info[path[i]][1]),
        #                                                    (road_node_info[path[i+1]][0]-road_node_info[path[i]][0]))])

        # return the allow action
        if len(allow_direction) == 0:
            print(robot_x, robot_y, last_local_goal_px, last_local_goal_py, next_local_goal_px, next_local_goal_py, simple_paths)
            print(self._env_done, self._global_goal_px, self._global_goal_py)
            raise Exception("None allow direction")

        for i in range(len(allow_direction)):
            if i + 1 < len(allow_direction):
                if allow_direction[i] == allow_direction[i + 1]:
                    continue

            if allow_action[1] != 1. and -math.pi / 4 < allow_direction[i][2] <= math.pi / 4:
                allow_action[1] = 1.
                next_station[1].append(allow_direction[i][0])
                next_station[1].append(allow_direction[i][1])
            elif allow_action[2] != 1. and math.pi / 4 < allow_direction[i][2] <= math.pi * 3 / 4:
                allow_action[2] = 1.
                next_station[2].append(allow_direction[i][0])
                next_station[2].append(allow_direction[i][1])
            elif allow_action[3] != 1. and ((math.pi * 3 / 4 < allow_direction[i][2] <= math.pi) or (
                    -math.pi < allow_direction[i][2] <= -math.pi * 3 / 4)):
                allow_action[3] = 1.
                next_station[3].append(allow_direction[i][0])
                next_station[3].append(allow_direction[i][1])
            elif allow_action[4] != 1. and -math.pi * 3 / 4 < allow_direction[i][2] <= -math.pi * 1 / 4:
                allow_action[4] = 1.
                next_station[4].append(allow_direction[i][0])
                next_station[4].append(allow_direction[i][1])

        return next_station, allow_action

        """
        for i in range(len(allow_direction)):
            if i + 1 < len(allow_direction):
                if allow_direction[i] == allow_direction[i + 1]:
                    continue

            if allow_action[1] != 1. and -math.pi / 8 < allow_direction[i][2] <= math.pi / 8:
                allow_action[1] = 1.
                next_station[1].append(allow_direction[i][0])
                next_station[1].append(allow_direction[i][1])
            elif allow_action[2] != 1. and math.pi / 8 < allow_direction[i][2] <= math.pi * 3 / 8:
                allow_action[2] = 1.
                next_station[2].append(allow_direction[i][0])
                next_station[2].append(allow_direction[i][1])
            elif allow_action[3] != 1. and math.pi * 3 / 8 < allow_direction[i][2] <= math.pi * 5 / 8:
                allow_action[3] = 1.
                next_station[3].append(allow_direction[i][0])
                next_station[3].append(allow_direction[i][1])
            elif allow_action[4] != 1. and math.pi * 5 / 8 < allow_direction[i][2] <= math.pi * 7 / 8:
                allow_action[4] = 1.
                next_station[4].append(allow_direction[i][0])
                next_station[4].append(allow_direction[i][1])
            elif allow_action[5] != 1. and ((math.pi * 7 / 8 < allow_direction[i][2] <= math.pi) or (
                    -math.pi < allow_direction[i][2] <= -math.pi * 7 / 8)):
                allow_action[5] = 1.
                next_station[5].append(allow_direction[i][0])
                next_station[5].append(allow_direction[i][1])
            elif allow_action[6] != 1. and -math.pi * 7 / 8 < allow_direction[i][2] <= -math.pi * 5 / 8:
                allow_action[6] = 1.
                next_station[6].append(allow_direction[i][0])
                next_station[6].append(allow_direction[i][1])
            elif allow_action[7] != 1. and -math.pi * 5 / 8 < allow_direction[i][2] <= -math.pi * 3 / 8:
                allow_action[7] = 1.
                next_station[7].append(allow_direction[i][0])
                next_station[7].append(allow_direction[i][1])
            elif allow_action[8] != 1. and -math.pi * 3 / 8 < allow_direction[i][2] <= -math.pi / 8:
                allow_action[8] = 1.
                next_station[8].append(allow_direction[i][0])
                next_station[8].append(allow_direction[i][1])

        return next_station, allow_action
        """

    """
    @staticmethod
    def search_allow_action(robot_x, robot_y, local_goal_x, local_goal_y, road_node_info, simple_paths):
        search the allow actions according to current position of agent and the paths from the agent to the goal
        # 初始化
        at_node = False
        allow_action = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        next_station = [[robot_x, robot_y], [], [], [], [], [], [], [], []]
        allow_direction = list()

        # 判断AGV是否位于站点         self._last_local_goal_px, self._last_local_goal_py,
        #                                                                         self._next_local_goal_px, self._next_local_goal_py,
        #                                                                         self.road_node_info, self.simple_paths
        coincide_node = None
        for path in simple_paths:
            for station in path:
                if math.sqrt((road_node_info[station][0]-robot_x)**2 + (road_node_info[station][1]-robot_y)**2) < 0.01:
                    at_node = True
                    coincide_node = station
                    break

        # 根据路径计算出所有许可的运动方向
        if at_node:
            for path in simple_paths:
                for i in range(len(path)-1):
                    if math.sqrt((road_node_info[coincide_node][0] - road_node_info[path[i]][0]) ** 2
                                 + (road_node_info[coincide_node][1] - road_node_info[path[i]][1]) ** 2) < 0.01:
                        allow_direction.append([road_node_info[path[i+1]][0], road_node_info[path[i+1]][1],
                                                math.atan2((road_node_info[path[i+1]][1]-road_node_info[path[i]][1]),
                                                           (road_node_info[path[i+1]][0]-road_node_info[path[i]][0]))])
        else:
            agent_line_dist = list()
            for path in simple_paths:
                for i in range(len(path)-1):
                    if road_node_info[path[i+1]][0] == local_goal_x and road_node_info[path[i+1]][1] == local_goal_y:

            """
    """
            for path in simple_paths:
                for i in range(len(path)-1):
                    cross = (road_node_info[path[i+1]][0] - road_node_info[path[i]][0]) * (robot_x - road_node_info[path[i]][0]) \
                            + (road_node_info[path[i+1]][1] - road_node_info[path[i]][1]) * (robot_y - road_node_info[path[i]][1])
                    if cross <= 0:
                        agent_line_dist.append([path[i], path[i+1], np.sqrt((robot_x - road_node_info[path[i]][0]) ** 2
                                                                            + (robot_y - road_node_info[path[i]][1]) ** 2)])
                        continue

                    d2 = (road_node_info[path[i+1]][0] - road_node_info[path[i]][0]) ** 2 \
                         + (road_node_info[path[i+1]][1] - road_node_info[path[i]][1]) ** 2
                    if cross >= d2:
                        agent_line_dist.append([path[i], path[i+1], np.sqrt((robot_x - road_node_info[path[i+1]][0]) ** 2
                                                                            + (robot_y - road_node_info[path[i+1]][1]) ** 2)])
                        continue
                    r = cross / d2
                    p0 = road_node_info[path[i]][0] + (road_node_info[path[i+1]][0] - road_node_info[path[i]][0]) * r
                    p1 = road_node_info[path[i]][1] + (road_node_info[path[i+1]][1] - road_node_info[path[i]][1]) * r
                    agent_line_dist.append([path[i], path[i+1], np.sqrt((robot_x - p0) ** 2 + (robot_y - p1) ** 2)])

                # find the nearest line index
                agent_line_dist_shortest = float("inf")
                agent_line_shortest_node_0 = None
                agent_line_shortest_node_1 = None

                for item in agent_line_dist:
                    if item[2] < agent_line_dist_shortest:
                        if(road_node_info[item[0]][0] == local_goal_x and road_node_info[item[0]][1] == local_goal_y) \
                                or (road_node_info[item[1]][0] == local_goal_x and road_node_info[item[1]][1] == local_goal_y):
                            agent_line_shortest_node_0 = item[0]
                            agent_line_shortest_node_1 = item[1]
                            agent_line_dist_shortest = item[2]

                allow_direction.append([road_node_info[agent_line_shortest_node_1][0], road_node_info[agent_line_shortest_node_1][1],
                                        math.atan2((road_node_info[agent_line_shortest_node_1][1] - road_node_info[agent_line_shortest_node_0][1]),
                                        (road_node_info[agent_line_shortest_node_1][0] - road_node_info[agent_line_shortest_node_0][0]))])
                """
    """
        # return the allow action
        if len(allow_direction) == 0:
            raise Exception("None allow direction")

        for i in range(len(allow_direction)):
            if i+1 < len(allow_direction):
                if allow_direction[i] == allow_direction[i+1]:
                    continue

            if allow_action[1] != 1. and -math.pi/8 < allow_direction[i][2] <= math.pi/8:
                allow_action[1] = 1.
                next_station[1].append(allow_direction[i][0])
                next_station[1].append(allow_direction[i][1])
            elif allow_action[2] != 1. and math.pi/8 < allow_direction[i][2] <= math.pi*3/8:
                allow_action[2] = 1.
                next_station[2].append(allow_direction[i][0])
                next_station[2].append(allow_direction[i][1])
            elif allow_action[3] != 1. and math.pi*3/8 < allow_direction[i][2] <= math.pi*5/8:
                allow_action[3] = 1.
                next_station[3].append(allow_direction[i][0])
                next_station[3].append(allow_direction[i][1])
            elif allow_action[4] != 1. and math.pi*5/8 < allow_direction[i][2] <= math.pi*7/8:
                allow_action[4] = 1.
                next_station[4].append(allow_direction[i][0])
                next_station[4].append(allow_direction[i][1])
            elif allow_action[5] != 1. and ((math.pi*7/8 < allow_direction[i][2] <= math.pi) or (-math.pi < allow_direction[i][2] <= -math.pi*7/8)):
                allow_action[5] = 1.
                next_station[5].append(allow_direction[i][0])
                next_station[5].append(allow_direction[i][1])
            elif allow_action[6] != 1. and -math.pi*7/8 < allow_direction[i][2] <= -math.pi*5/8:
                allow_action[6] = 1.
                next_station[6].append(allow_direction[i][0])
                next_station[6].append(allow_direction[i][1])
            elif allow_action[7] != 1. and -math.pi*5/8 < allow_direction[i][2] <= -math.pi*3/8:
                allow_action[7] = 1.
                next_station[7].append(allow_direction[i][0])
                next_station[7].append(allow_direction[i][1])
            elif allow_action[8] != 1. and -math.pi*3/8 < allow_direction[i][2] <= -math.pi/8:
                allow_action[8] = 1.
                next_station[8].append(allow_direction[i][0])
                next_station[8].append(allow_direction[i][1])

        return next_station, allow_action
    """

    # @staticmethod
    def search_next_station(self, last_local_goal_px, last_local_goal_py, road_node_info, simple_paths):
        """
        search the next station according to current position of agent and the paths from the agent to the goal
        """
        # 初始化
        next_station = []

        # 查找重合站点的编号
        coincide_node = None
        # for path in simple_paths:
        #     for station in path:
        #         if math.sqrt((road_node_info[station][0] - last_local_goal_px) ** 2 + (road_node_info[station][1] - last_local_goal_py) ** 2) < 0.01:
        #             coincide_node = station
        #             break

        for station in simple_paths:
            if math.sqrt((road_node_info[station][0] - last_local_goal_px) ** 2 + (road_node_info[station][1] - last_local_goal_py) ** 2) < 0.01:
                coincide_node = station
                break
        # 在简单路径中，查找站点的编号，返回站点的坐标
        # for path in simple_paths:
        #     for i in range(len(path) - 1):
        #         if math.sqrt((road_node_info[coincide_node][0] - road_node_info[path[i]][0]) ** 2
        #                      + (road_node_info[coincide_node][1] - road_node_info[path[i]][1]) ** 2) < 0.01:
        #             next_station.append(road_node_info[path[i + 1]][0])
        #             next_station.append(road_node_info[path[i + 1]][1])
        for i in range(len(simple_paths) - 1):
            if math.sqrt((road_node_info[coincide_node][0] - road_node_info[simple_paths[i]][0]) ** 2
                         + (road_node_info[coincide_node][1] - road_node_info[simple_paths[i]][1]) ** 2) < 0.01:
                next_station.append(road_node_info[simple_paths[i + 1]][0])
                next_station.append(road_node_info[simple_paths[i + 1]][1])

        return next_station

    def simplify_paths(self, simple_paths, last_station):
        # 删除没有该站点的路径
        for i in range(len(simple_paths)-1, -1, -1):
            if last_station == simple_paths[i][0] or last_station == simple_paths[i][1]:
                pass
            else:
                simple_paths.remove(simple_paths[i])

        # 截取该站点及目标的路径
        for path in simple_paths:
            for i in range(len(path)-1, -1, -1):
                if i < path.index(last_station):
                    path.remove(path[i])
                else:
                    pass
        return simple_paths

    def check_path_deadlock(self, simple_paths, other_agents):
        # 判断2辆车相互之间的所有路径是否相反，若全部相反则判断为死锁
        for other_agent in other_agents:
            other_agent_path_deadlock = []
            for path in simple_paths:
                for other_agent_path in other_agent.simple_paths:
                    if len(other_agent_path) == 1:
                        other_agent_path_deadlock.append(0)
                    else:
                        # other_agent_path = list(reversed(path))
                        if (self.check_line_in_path([other_agent_path[1], other_agent_path[0]], path) and
                                self.check_line_in_path([path[1], path[0]], other_agent_path)):
                            if path[0:path.index(other_agent_path[0])+1] == list(reversed(other_agent_path[0:other_agent_path.index(path[0])+1])):
                                other_agent_path_deadlock.append(1)
                            else:
                                other_agent_path_deadlock.append(0)
                        else:
                            other_agent_path_deadlock.append(0)

            if len(simple_paths)*len(other_agent.simple_paths) == sum(other_agent_path_deadlock):
                return True
        return False

    def check_terminal_deadlock(self, simple_paths, other_agents):
        # 判断2辆车相互之间的终点是否由于相同终点导致死锁。
        self.G.add_edges_from(self.node_edges)
        for other_agent in other_agents:
            other_agent_path_deadlock = []
            for path in simple_paths:
                for other_agent_path in other_agent.simple_paths:
                    if path[-1] == other_agent_path[-1]:
                        if self.check_line_in_path([other_agent_path[0], other_agent_path[1]], path):
                            terminal_path = nx.astar_path(self.G, source=other_agent_path[-1], target=path[0], weight='len')
                            if list(reversed(path)) == terminal_path[0:len(path)+1]:
                                other_agent_path_deadlock.append(1)
                            else:
                                other_agent_path_deadlock.append(0)
                        else:
                            other_agent_path_deadlock.append(0)
                    else:
                        other_agent_path_deadlock.append(0)
            if len(simple_paths)*len(other_agent.simple_paths) == sum(other_agent_path_deadlock):
                self.G.clear()
                return True
        self.G.clear()
        return False

    def select_optimal_path(self, simple_paths, other_agents):
        optimal_path_index = []
        if len(simple_paths) == 1:
            optimal_path_index.append(0)
        else:
            self_shortest_path = simple_paths[0]
            for other_agent in other_agents:
                other_shortest_path = other_agent.simple_paths[0]
                if (self.check_line_in_path([other_shortest_path[1], other_shortest_path[0]], self_shortest_path) and
                    self.check_line_in_path([self_shortest_path[1], self_shortest_path[0]], other_shortest_path)):
                    if (self_shortest_path[0:self_shortest_path.index(other_shortest_path[0]) + 1] ==
                            list(reversed(other_shortest_path[0:other_shortest_path.index(self_shortest_path[0]) + 1]))):
                        if len(other_agent.simple_paths) == 1:
                            optimal_path_index.append(1)
                        else:
                            if other_shortest_path[0] in simple_paths[1]:
                                self_2nd_path_cost = self.calculate_path_cost(simple_paths[1], 0, simple_paths[1].index(other_shortest_path[0])+1)
                            else:
                                self_2nd_path_cost = 0
                            if self_shortest_path[0] in other_agent.simple_paths[1]:
                                other_2nd_path_cost = self.calculate_path_cost(other_agent.simple_paths[1], 0, other_agent.simple_paths[1].index(self_shortest_path[0])+1)
                            else:
                                other_2nd_path_cost = 0

                            if self_2nd_path_cost < other_2nd_path_cost:
                                optimal_path_index.append(1)
                            else:
                                optimal_path_index.append(0)
                    else:
                        optimal_path_index.append(0)
                else:
                    optimal_path_index.append(0)
        if sum(optimal_path_index) > 0:
            optimal_path = 1
        else:
            optimal_path = 0
        return optimal_path

    def calculate_path_cost(self, path, start_node, end_node):
        path_list = path[start_node:end_node]
        path_length = []
        for i in range(len(path_list)-1):
            for j in range(len(self.node_edges)):
                if self.node_edges[j][0] == path_list[i] and self.node_edges[j][1] == path_list[i+1]:
                    path_length.append(self.node_edges[j][2]['len'])
                    break
        path_cost = sum(path_length)
        return path_cost

    def check_collision_deadlock(self, robot_x, robot_y, v, last_goal_x, last_goal_y, next_goal_x, next_goal_y,
                                 other_agents):
        # 判断2辆车相互之间是否为了避免碰撞，形成了死锁状态
        other_agent_collision_deadlock = list()
        for other_agent in other_agents:
            dis_robot = math.sqrt((robot_x-other_agent.px)**2+(robot_y-other_agent.py)**2)
            if dis_robot <= 3 * GoConfig.STEPPING_VELOCITY:
                if ((last_goal_x == other_agent.last_local_gx and last_goal_y == other_agent.last_local_gy) or
                        (last_goal_x == other_agent.next_local_gx and last_goal_y == other_agent.next_local_gy) or
                        (next_goal_x == other_agent.last_local_gx and next_goal_y == other_agent.last_local_gy)):
                    collision_self = False
                    collision_other = False

                    dis_self = math.sqrt((next_goal_x - robot_x) ** 2 + (next_goal_y - robot_y) ** 2)
                    if dis_self > 0:
                        direction_x = (next_goal_x - robot_x) / dis_self
                        direction_y = (next_goal_y - robot_y) / dis_self
                        for i in range(20):
                            nextpos_x = robot_x + 0.1 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_x
                            nextpos_y = robot_y + 0.1 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_y
                            if math.sqrt((nextpos_x - other_agent.px) ** 2 + (nextpos_y - other_agent.py) ** 2) < 2 * GoConfig.RADIUS:
                                collision_self = True

                    dis_other = math.sqrt((other_agent.px - other_agent.next_local_gx) ** 2 + (other_agent.py - other_agent.next_local_gy) ** 2)
                    if dis_other > 0:
                        direction_other_x = (other_agent.next_local_gx - other_agent.px) / dis_other
                        direction_other_y = (other_agent.next_local_gy - other_agent.py) / dis_other
                        for i in range(20):
                            nextpos_other_x = other_agent.px + 0.1 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_other_x
                            nextpos_other_y = other_agent.py + 0.1 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_other_y
                            if math.sqrt((nextpos_other_x - robot_x) ** 2 + (nextpos_other_y - robot_y) ** 2) < 2 * GoConfig.RADIUS:
                                collision_other = True

                    if collision_self and collision_other:
                        other_agent_collision_deadlock.append(1)
                    else:
                        other_agent_collision_deadlock.append(0)
                else:
                    # velocity mean
                    for i in range(1, self.v_mean.shape[0]):
                        self.v_mean[i-1] = self.v_mean[i]
                    self.v_mean[-1] = v
                    v_mean = np.mean(self.v_mean)

                    if v_mean == 0 and other_agent.v_mean == 0:
                        other_agent_collision_deadlock.append(1)
                    else:
                        other_agent_collision_deadlock.append(0)
            else:
                other_agent_collision_deadlock.append(0)

        if sum(other_agent_collision_deadlock) >= 1:
            return True
        else:
            return False

    """
    if v == 0 and math.sqrt(other_agent.vx ** 2 + other_agent.vy ** 2) == 0:
        collision_self = False
        collision_other = False
        collision_together = False
        collision_deadlock = False

        dis_self = math.sqrt((next_goal_x - robot_x) ** 2 + (next_goal_y - robot_y) ** 2)
        direction_x = (next_goal_x - robot_x) / dis_self
        direction_y = (next_goal_y - robot_y) / dis_self
        for i in range(20):
            nextpos_x = robot_x + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_x
            nextpos_y = robot_y + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_y
            if math.sqrt((nextpos_x - other_agent.px) ** 2 + (nextpos_y - other_agent.py) ** 2) < 2 * GoConfig.RADIUS:
                collision_self = True

        dis_other = math.sqrt(
            (other_agent.px - other_agent.next_local_gx) ** 2 + (other_agent.py - other_agent.next_local_gy) ** 2)
        direction_other_x = (other_agent.next_local_gx - other_agent.px) / dis_other
        direction_other_y = (other_agent.next_local_gy - other_agent.py) / dis_other
        for i in range(20):
            nextpos_other_x = other_agent.px + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_other_x
            nextpos_other_y = other_agent.py + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_other_y
            if math.sqrt((nextpos_other_x - robot_x) ** 2 + (nextpos_other_y - robot_y) ** 2) < 2 * GoConfig.RADIUS:
                collision_other = True

        for i in range(20):
            nextpos_x = robot_x + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_x
            nextpos_y = robot_y + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_y
            nextpos_other_x = other_agent.px + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_other_x
            nextpos_other_y = other_agent.py + 0.05 * (i + 1) * GoConfig.STEPPING_VELOCITY * direction_other_y
            if math.sqrt((nextpos_other_x - nextpos_x) ** 2 + (nextpos_other_y - nextpos_y) ** 2) < 2 * GoConfig.RADIUS:
                collision_together = True

        if collision_self and collision_other and collision_together:
            collision_deadlock = True

        if collision_deadlock:
            other_agent_collision_deadlock.append(1)
        else:
            other_agent_collision_deadlock.append(0)
    else:
        other_agent_collision_deadlock.append(0)
    """

    @staticmethod
    def check_line_in_path(line, path):
        if (line[0] in path) and (line[1] in path):
            if path.index(line[1]) - path.index(line[0]) == 1:
                return True
        else:
            return False

    @staticmethod
    def search_route_action(current_goal_x, current_goal_y, last_goal_x, last_goal_y):
        route_action = [1, 0.1, 0.1, 0.1, 0.1]
        route_angle = math.atan2((current_goal_y - last_goal_y), (current_goal_x - last_goal_x))
        if route_angle == math.pi / 2:
            route_action[1] = 1
        elif route_angle == - math.pi / 2:
            route_action[2] = 1
        elif route_angle == 0:
            route_action[3] = 1
        elif route_angle == math.pi:
            route_action[4] = 1
        else:
            raise AttributeError("None route action")
        return route_action

    def optimal_path_action(self, robot_x, robot_y, goal_x, goal_y, G, road_node_Nos, road_node_info,
                                    road_lines, road_lines_num, node_edges):
        # start = time.time()
        # add target node
        target_node_coordinate = np.zeros((1, 2))
        target_node_coordinate[0][0] = int(goal_x / 0.05)
        target_node_coordinate[0][1] = int(goal_y / 0.05)
        target_node = None
        for (key, value) in road_node_info.items():
            if abs(value[0]) < abs(target_node_coordinate[0][0]) + 1 and \
                    abs(value[1]) < abs(target_node_coordinate[0][1]) + 1:
                target_node = key

        if target_node == 0:
            print(target_node)
            raise Exception("wrong target node", target_node)

        # add agent node
        agent_node_No = 0
        agent_node_coordinate = np.zeros((1, 2))
        agent_node_coordinate[0][0] = int(robot_x / 0.05)
        agent_node_coordinate[0][1] = int(robot_y / 0.05)
        agent_node = dict(zip([agent_node_No], agent_node_coordinate))
        road_node_info.update(agent_node)
        env_node_info = road_node_info

        # add node
        env_node_Nos = [agent_node_No] + road_node_Nos
        G.add_nodes_from(env_node_Nos)

        # add edges from agent to the nearest road line
        # calculate the distance from the agent to the lines
        env_node_labels = dict(zip(env_node_Nos, env_node_Nos))

        # agent_line_dist = []
        # for i in range(road_lines_num):
        #     dist = self.distance_p2seg(road_lines[i][0], road_lines[i][1], road_lines[i][2], road_lines[i][3],
        #                                agent_node_coordinate[0][0], agent_node_coordinate[0][1])
        #     agent_line_dist.append(dist)
        #
        # # find the nearest line index
        # agent_line_dist_shortest = float("inf")
        # agent_line_shortest_index = 0
        # for index, item in enumerate(agent_line_dist):
        #     if item < agent_line_dist_shortest:
        #         agent_line_shortest_index = index
        #         agent_line_dist_shortest = item

        agent_line_shortest_index = self.distance_p2network(road_lines, agent_node_coordinate[0][0], agent_node_coordinate[0][1])

        # find the shortest line's node
        agent_line_shortest_node0 = None
        agent_line_shortest_node1 = None
        for (key, value) in road_node_info.items():
            if value[0] == road_lines[agent_line_shortest_index][0] and value[1] == \
                    road_lines[agent_line_shortest_index][1]:
                agent_line_shortest_node0 = key
            if value[0] == road_lines[agent_line_shortest_index][2] and value[1] == \
                    road_lines[agent_line_shortest_index][3]:
                agent_line_shortest_node1 = key
        # end = time.time()
        # print("-----------------------shortest path action time--------------------------------------", start - end)

        # start = time.time()
        # add new edges from the agent node to road note
        # node_edges.append([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
        #     (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
        #                 road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
        node_edges.append([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])

        G.add_edges_from(node_edges)

        # The robot is at the road node or not
        at_node = False
        coincide_node = None
        for (key, value) in road_node_info.items():
            if key == 0:
                continue
            if value[0] == agent_node_coordinate[0][0] and value[1] == agent_node_coordinate[0][1]:
                at_node = True
                coincide_node = key

        shortest_path_direction = []
        shortest_path_action = [1, 0.1, 0.1, 0.1, 0.1]

        # start = time.time()
        if at_node:
            all_shortest_paths = nx.all_shortest_paths(G, source=coincide_node, target=target_node, weight='len')
            shortest_path_length = nx.shortest_path_length(G, source=coincide_node, target=target_node,
                                                           weight='len')
            for shortest_path in all_shortest_paths:
                if len(shortest_path) >= 2:
                    shortest_path_direction.append(
                        math.atan2((road_node_info[shortest_path[1]][1] - road_node_info[shortest_path[0]][1]),
                                   (road_node_info[shortest_path[1]][0] - road_node_info[shortest_path[0]][0])))
                else:
                    continue
        else:
            all_shortest_paths = nx.all_shortest_paths(G, source=agent_node_No, target=target_node, weight='len')
            shortest_path_length = nx.shortest_path_length(G, source=agent_node_No, target=target_node, weight='len')

            for shortest_path in all_shortest_paths:
                    shortest_path_direction.append(
                        math.atan2((road_node_info[shortest_path[1]][1] - agent_node_coordinate[0][1]),
                                   (road_node_info[shortest_path[1]][0] - agent_node_coordinate[0][0])))

        # # return the shortest path action
        # if len(shortest_path_direction) == 0:
        #     raise Exception("None shortest path direction")

        for i in shortest_path_direction:
            if math.pi / 2 - 0.01 <= i <= math.pi / 2 + 0.01:
                shortest_path_action[1] = 1
            elif - math.pi / 2 - 0.01 <= i <= - math.pi / 2 + 0.01:
                shortest_path_action[2] = 1
            elif 0 - 0.01 <= i <= 0 + 0.01:
                shortest_path_action[3] = 1
            elif math.pi - 0.01 <= i <= math.pi + 0.01:
                shortest_path_action[4] = 1
            else:
                raise AttributeError("None shortest path direction")

        # node_edges.remove([agent_node_No, agent_line_shortest_node0, {'len': np.sqrt(
        #     (road_node_info[agent_line_shortest_node0][0] - agent_node_coordinate[0][0]) ** 2 + (
        #                 road_node_info[agent_line_shortest_node0][1] - agent_node_coordinate[0][1]) ** 2)}])
        node_edges.remove([agent_node_No, agent_line_shortest_node1, {'len': np.sqrt(
            (road_node_info[agent_line_shortest_node1][0] - agent_node_coordinate[0][0]) ** 2 + (
                        road_node_info[agent_line_shortest_node1][1] - agent_node_coordinate[0][1]) ** 2)}])

        G.clear()
        # end = time.time()
        # print("-----------------------shortest path action time--------------------------------------", start - end)
        return shortest_path_action, shortest_path_length * 0.05

    def close(self):
        if self.enable_show is True:
            self.emulator.gnuplot_exit()

    @staticmethod
    @jit
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
    @jit(nopython=True)
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
