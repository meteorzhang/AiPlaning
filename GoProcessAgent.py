# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value
# from threading import Thread
# from queue import Queue
import numpy as np
import time
import math
import sys
import tracemalloc
from GoConfig import GoConfig
from Experience import Experience
from GoEnvironment import GoEnvironment
import copy
from GoStatus import AgentState
import networkx as nx

class GoProcessAgent(Process):
    def __init__(self, id, shared_env, actions, prediction_q, training_q, episode_log_q, enable_show, episode_count,road_node_Nos, road_node_info, road_lines,road_directions, road_lines_num, node_edges):
        super(GoProcessAgent, self).__init__()

        self.id = str(id)
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q
        self.share_env = shared_env
        self._actions = actions
        self.num_actions = self._actions.get_actions_nums()
        self.actions = np.arange(self.num_actions)
        self.enable_show = enable_show
        self.env = GoEnvironment(id, self.share_env, actions, self.enable_show)
        self.con = self.share_env.agents_con
        self.discount_factor = GoConfig.DISCOUNT
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)
        # self.exit_flag = 0
        self.episode_count = episode_count
        self.epsilon = 0.5
        self.epsilon_start = 0.5
        self.epsilon_end = 0.9  # 0.9
        self.flag = False
        self.parking_time = 0
        self.G = nx.DiGraph()
        self.road_node_Nos = road_node_Nos
        self.road_node_info = road_node_info
        self.road_lines = road_lines
        self.road_directions=road_directions
        self.road_lines_num = road_lines_num
        self.node_edges = node_edges



    def init_data(self, id, x, y, goal_x, goal_y):
        # print("x: ", x, " y:", y)
        self.__id = id
        # self.x, self.y = self.__world_coordinate(x, y)
        # self.goal_x, self.goal_y = self.__world_coordinate(goal_x, goal_y)
        # add
        self._start_x = x
        self._start_y = y

        self._goal_x = goal_x
        self._goal_y = goal_y

        self.v_mean = np.array([1e-5, 1e-5, 1e-5])

        self.shortest_path_action = [1, 0.1, 0.1, 0.1, 0.1]

        self.shortest_path_length = 0

        self.state = 0



        curr_status = AgentState(self._start_x, self._start_y, 0, 0, self._goal_x, self._goal_y, self.radius,
                            np.mean(self.v_mean), self.shortest_path_action, self.shortest_path_length, self.__id, 0)

        self.share_env.update_agent_status(curr_status)  # 共享位姿
        print("初始化时，共享数据时间：", time.time()-s)
    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, GoConfig.REWARD_MIN, GoConfig.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        # return experiences[:-1]
        return experiences

    def convert_data(self, experiences):
        x_ = [exp.state for exp in experiences]
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def predict(self, full_state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, full_state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def set_goal_xy(self,x, y, goal_x, goal_y):
        self._robot_px = x
        self._robot_py = y
        self._global_goal_px = goal_x
        self._global_goal_py = goal_y
        s
        self.has_path = False
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

        self.share_env.update_agent_status(status)
    def select_action(self, prediction):
        if GoConfig.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

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
                        if self.check_line_in_path([other_agent_path[1], other_agent_path[0]], path) and \
                                self.check_line_in_path([path[1], path[0]], other_agent_path):
                            if path[0:path.index(other_agent_path[0])+1] == list(reversed(other_agent_path[0:other_agent_path.index(path[0])+1])):
                                other_agent_path_deadlock.append(1)
                            else:
                                other_agent_path_deadlock.append(0)
                        elif len(other_agent_path) > 2 and self.check_line_in_path([other_agent_path[2], other_agent_path[1]], path):
                            if path[1] == other_agent_path[2] and path[0] not in other_agent_path:
                                other_agent_path_deadlock.append(1)
                            else:
                                other_agent_path_deadlock.append(0)
                        else:
                            other_agent_path_deadlock.append(0)

            if len(simple_paths)*len(other_agent.simple_paths) == sum(other_agent_path_deadlock):
                return True
        return False

    def check_terminal_deadlock(self, simple_paths, other_agents):
        # 判断2辆车相互之间,由于终点造成的，若全部相反则判断为死锁
        for other_agent in other_agents:
            other_agent_path_deadlock = []
            for path in simple_paths:
                for other_agent_path in other_agent.simple_paths:
                    if len(other_agent_path) == 1:
                        other_agent_path_deadlock.append(0)
                    else:
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

    @staticmethod
    def check_line_in_path(line, path):
        if (line[0] in path) and (line[1] in path):
            if path.index(line[1]) - path.index(line[0]) == 1:
                return True
        else:
            return False

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []
        time_count = 0
        reward_sum = 0.0
        step_iteration = 0

        while not done:
            current_full_state, other_agent_state = self.env.get_full_current_state()

            # if current_full_state is None:
            #     self.env.step(None, other_agent_state)
            #     continue

            # # 生成在终点处的随机停车时间
            # if current_full_state.status == 3:
            #     self.parking_time = 10 + np.random.randint(0, 10)
            #
            # # 在停车时间内，不进行预测和训练，停车，但更新状态。
            # if self.parking_time != 0:
            #     self.parking_time = self.parking_time - 1
            #     self.env.step(None, other_agent_state)
            #     time.sleep(0.05)
            #     self.con.notify()
            #     self.con.wait()
            #     continue

            prediction, value = self.predict(current_full_state)
            action = self.select_action(prediction)

            # 若执行动作将发生碰撞，则选择静止
            if len(current_full_state.other) > 0:
                a_v = self._actions.get_action_by_indx(action)
                # 到下一个站点的距离
                d_next = math.sqrt((current_full_state.self[0] - current_full_state.self[6])**2
                                   + (current_full_state.self[1] - current_full_state.self[7])**2)

                for i in range(20):
                    if d_next <= 0.1 * (i+1) * a_v:
                        next_pos_x = current_full_state.self[6]
                        next_pos_y = current_full_state.self[7]
                    else:
                        next_pos_x = current_full_state.self[0] + 0.1 * (i+1) * a_v * (current_full_state.self[6] - current_full_state.self[0]) / d_next
                        next_pos_y = current_full_state.self[1] + 0.1 * (i+1) * a_v * (current_full_state.self[7] - current_full_state.self[1]) / d_next

                    d_other = math.sqrt((next_pos_x - current_full_state.other[-1][1])**2 + (next_pos_y - current_full_state.other[-1][2])**2)

                    if d_other <= current_full_state.other[-1][11]:
                        action = 0

                if len(current_full_state.path) > 2:
                    d_other_ = math.sqrt((current_full_state.path[-3][0] - current_full_state.other[-1][1]) ** 2 + (current_full_state.path[-3][1] - current_full_state.other[-1][2]) ** 2)
                    if d_other_ <= current_full_state.other[-1][11]:
                        action = 0

                # 针对潜在的碰撞死锁情况，尝试解锁的动作
                # 终点一样的情形下，距离近的优先选择
                if (action == 0 and current_full_state.self[6] == current_full_state.other[-1][7]
                        and current_full_state.self[7] == current_full_state.other[-1][8]):
                    d_next_other = math.sqrt((current_full_state.other[-1][1] - current_full_state.other[-1][7]) ** 2
                                             + (current_full_state.other[-1][2] - current_full_state.other[-1][8]) ** 2)
                    if d_next <= d_next_other:
                        if GoConfig.PLAY_MODE:
                            action = np.argmax(prediction)
                        else:
                            action = np.random.choice(self.actions, p=prediction)
                # 起点为对方终点的情形下，起点AGV优先选择
                if (action == 0 and current_full_state.self[4] == current_full_state.other[-1][7]
                        and current_full_state.self[5] == current_full_state.other[-1][8]):
                    if GoConfig.PLAY_MODE:
                        action = np.argmax(prediction)
                    else:
                        action = np.random.choice(self.actions, p=prediction)

                # 双方所处线段没有相关性的情形
                self_line = [[current_full_state.self[4], current_full_state.self[5]], [current_full_state.self[6], current_full_state.self[7]]]
                other_line = [[current_full_state.other[-1][5], current_full_state.other[-1][6]], [current_full_state.other[-1][7], current_full_state.other[-1][8]]]
                count = 0
                for i in range(len(self_line)):
                    for j in range(len(other_line)):
                        if self_line[i] == other_line[j]:
                            count = count + 1
                if action == 0 and count == 0:
                    if GoConfig.PLAY_MODE:
                        action = np.argmax(prediction)
                    else:
                        action = np.random.choice(self.actions, p=prediction)

            # 若执行动作将发生路径死锁，则选择静止
            if len(current_full_state.other) > 0 and action != 0:
                a_v = self._actions.get_action_by_indx(action)
                # 到下一个站点的距离
                d_next = math.sqrt((current_full_state.self[0] - current_full_state.self[6])**2
                                   + (current_full_state.self[1] - current_full_state.self[7])**2)
                # 对下一个站点的死锁判断
                if d_next < 3 * a_v:
                    self_simple_paths = copy.deepcopy(self.share_env.agents_dynamic_status.get(int(self.id)).simple_paths)
                    for path in self_simple_paths:
                        path.remove(path[0])
                else:
                    self_simple_paths = copy.deepcopy(self.share_env.agents_dynamic_status.get(int(self.id)).simple_paths)

                if self.check_path_deadlock(self_simple_paths, other_agent_state):
                    action = 0

                if d_next < 3 * a_v and action != 0 and len(current_full_state.path) > 2:
                    for path in self_simple_paths:
                        path.remove(path[0])

                    if self.check_path_deadlock(self_simple_paths, other_agent_state):
                        action = 0

            # if action != 0:
            #     self_status = self.share_env.agents_dynamic_status.get(int(self.id))
            #     other_status = self.share_env.get_other_agents_status(int(self.id))
            #     if self.check_deadlock(self_status.simple_paths, other_status):
            #         action = 0

            # # 根据动作选择站点
            # if action == 0:
            #     stations = []
            #     for index, item in enumerate(current_full_state.observation[1:], start=1):
            #         if item == 1:
            #             stations.append(index)
            #     station = np.random.choice(np.array(stations))
            # else:
            #     station = action

            # 执行动作和站点，并计算奖励
            # print("action2", action)
            reward, done, forced_stop = self.env.step(action, other_agent_state)
            reward_sum += reward

            if forced_stop == 1:
                action = 0

            if GoConfig.MAX_STEP_ITERATION < step_iteration:
                step_iteration = 0
                done = True

            exp = Experience(current_full_state, action, prediction, reward, done)

            experiences.append(exp)

            # self_current_state = self.env.get_self_current_state()
            # if done or time_count == GoConfig.TIME_MAX or self_current_state.mission_status == 3:

            if done or time_count == GoConfig.TIME_MAX:
                terminal_reward = 0 if done else value
                updated_exps = GoProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)

                x_, r_, a_ = self.convert_data(updated_exps)
                # 产生一次返回值继续执行下面
                yield x_, r_, a_, reward_sum
                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            step_iteration += 1
            time_count += 1
            # time.sleep(0.05)
            self.con.notify()
            self.con.wait()

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(1)
        np.random.seed(np.int32(time.time() % 1 * 1000 + int(self.id) * 10))
        # tracemalloc.start()
        self.con.acquire()

        while self.exit_flag == 0:
            total_reward = 0
            total_length = 0
            # snapshot1 = tracemalloc.take_snapshot()

            epsilon_multiplier = (self.epsilon_end - self.epsilon_start) / GoConfig.ANNEALING_EPISODE_COUNT
            step = min(self.episode_count.value, GoConfig.ANNEALING_EPISODE_COUNT - 1)
            self.epsilon = self.epsilon_start + epsilon_multiplier * step

            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                # her_agent=env.get_other_agents_status(self.id)
                # maps=env.map_pic_array
                # x_={}
                self.training_q.put((x_, r_, a_))
            # snapshot2 = tracemalloc.take_snapshot()
            # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            # topn = 5
            # print("[ Top {} ]".format(topn))
            # for stat in top_stats[:topn]:
            #     print(stat)
            #
            # top_stats1 = snapshot2.statistics('traceback')
            # pick the biggest memory block
            # stat1 = top_stats1[0]
            # print("%s memory blocks: %.1f KiB" % (stat1.count, stat1.size / 1024))
            # for line in stat1.traceback.format():
            #     print(line)
            # if self.episode_count.value < GoConfig.EPISODES / 2:
            #     self.epsilon = 2/3
            # else:
            #     self.epsilon = 1

            self.episode_log_q.put((datetime.now(), total_reward, total_length))

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