from multiprocessing import Queue
import sys
#sys.path.append('../')

from GoConfig import GoConfig
from ThreadPredictor import ThreadPredictor
#from threading import Thread
from GoProcessAgent import GoProcessAgent
from GoNetwork import GoNetwork
from GoAgentAction import GoAgentAction
from GoSharedStatus import GoSharedStatus
from PyQt5.QtCore import *
import numpy as np
import time

class Navigator1(object):

    def __init__(self):
        pass

        # if GoConfig.LOAD_CHECKPOINT:  # 加载模型
        #     self.stats.episode_count.value = self.model.load()
        # self.model = None

    def init(self):
        self.map_m = None
        self.agents = dict()  # AGENT字典
        self.isready = False  # indicate the Network loaded or didn't


        self.share = GoSharedStatus()

        self.prediction_q = Queue(maxsize=GoConfig.MAX_QUEUE_SIZE)  # 预测队列

        self.actions = GoAgentAction()

        self.model = GoNetwork(GoConfig.DEVICE, GoConfig.NETWORK_NAME, self.actions.action_space_nums)

        if GoConfig.LOAD_CHECKPOINT:
            self.model.load()




        # Build the directed road network from the road network file
        #self.route_palnning = Route_planning(path)

        self.predictors = []
        for i in range(8):
            self.add_predictor()  # 预测线程

    def get_all_state(self):

        # 执行一次循环
        # for d, agent in self.agents.items():
        #     agent.run_agent()
        return self.agents[0].share_env.get_all_state()


    def init_data(self, map_path, car_list):
        self._init_road_network(map_path)
        self.agvs = car_list
        self.update_agents(self.agvs)  # 更新carlist ,启动每个线程

    def delete(self):  # 销毁预测线程
        for i in range(10):
            self.remove_predictor()

    def loadmap(self, map_url):  # 初始化,如何应对地图大小的问题
        self.map_path = map_url
        pass



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

    def update_agents(self, car_list):
        # print("update_agents!!!")
        self.road_node_Nos, self.road_node_info, self.road_lines, self.road_directions,self.road_lines_num, self.node_edges = self._init_road_network(
            self.share.road_url)
        for x in car_list:
            self.add_agent(x,self.road_node_Nos, self.road_node_info, self.road_lines, self.road_directions,self.road_lines_num, self.node_edges)
        self.start_all_agent()  # 开始所有agent线程

    # 添加新的agent
    def add_agent(self, a,road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges):  # 为每个agent创建


        agent = GoProcessAgent(a[0], self.share, self.actions, self.prediction_q,road_node_Nos, road_node_info, road_lines, road_lines_num, node_edges)


        agent.init_data(a[0], a.x, a.y, a.x, a.y)


        self.agents[a[0]] = agent

    def set_goal(self, agv_id,stationcode,x,y,goal_x,goal_y):


        self.agents[agv_id].set_goal_xy(x, y, goal_x, goal_y)





    def delete_agent(self,x): # 删除
        agent=self.agents[x]
        #if agent is not None and isinstance(agent,GoProcessAgent):
        agent.stop()


        pass
    def delete_all_agent(self):
        for (k, v) in self.agents.items():
            self.delete_agent(k)

    def pause_agent(self,id):  # 暂停
        agent = self.agents[id]
        if agent is not None and isinstance(agent, GoProcessAgent):
            agent.pause()

    def pause_all_agent(self,id):  # 暂停所有AGENT
        for (k, v) in self.agents.items():
            self.pause_agent(k)
        pass

    def enable_show(self, status):  # 使能显示端
        if status:
            pass

    def start_all_agent(self):
        for (k, v) in self.agents.items():
            self.start_agent(k)
        pass

    def start_agent(self, fid):
        agent = self.agents[fid]
        #print("start:", id)
        if agent is not None and isinstance(agent, GoProcessAgent):
            agent.resume()  # 等待任务状态
            agent.start()  # 启动线程

    def get_agent_by_id(self, fid):
        return self.agents[fid]

    def add_predictor(self):  # 增加预测线程
        print("增加预测线程", len(self.predictors))
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):  # 减少预测线程
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()


    def updateAGV(self,agvs):
        self.agvs = agvs
