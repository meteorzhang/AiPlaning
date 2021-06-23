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


class GoProcessAgent(Process):
    def __init__(self, id, shared_env, actions, prediction_q, training_q, episode_log_q, enable_show, episode_count):
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
