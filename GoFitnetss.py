from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time
import math
import sys
from math import sqrt
from GoConfig import GoConfig


class GoFitness:
    def __init__(self):
        self._env_robot_x_last = 0.0
        self._env_robot_y_last = 0.0
        self._env_robot_orientation_last = 0.0
        self.last_optimal_path_len = 0.0
        self._collision_penalty = GoConfig.COLLISION_PENALTY
        self._reach_goal_reward = GoConfig.REACH_GOAL_REWARD
        self._approach_goal_reward = GoConfig.APPROACH_GOAL_REWARD
        self._deviate_road_penalty = GoConfig.DEVIATE_ROAD_PENALTY
        self._other_agv_distance_threshold = GoConfig.OTHER_AGV_DISTANCE_THRESHOLD
        self._allow_area_distance_threshold = GoConfig.ALLOW_AREA_DISTANCE_THRESHOLD
        self._road_network_distance_threshold = GoConfig.ROAD_NETWORK_DISTANCE_THRESHOLD
        self._min_factor = GoConfig.MIN_FACTOR
        self._max_factor = GoConfig.MAX_FACTOR
        self._last_agents_average_velocity = 1e-5  # last agents average velocity

    def calculate_reward(self, env_robot_x, env_robot_y, env_robot_orientation, last_goal_x, last_goal_y, env_goal_x,
                         env_goal_y, env_done, nearest_road_dis, nearest_allow_area_dis, nearest_other_agent_dis,
                         v_mean):
        """
        Calculate the reward for one step
        :param env_robot_x:
        :param env_robot_y:
        :param env_robot_orientation:
        :param env_goal_x:
        :param env_goal_y:
        :param env_done:
        :param nearest_road_dis:
        :param nearest_allow_area_dis:
        :param nearest_other_agent_dis:
        :return:
        Return reward and done
        """

        done = False

        # reward for goal
        if env_done == 0:
            # ---------before--------------
            # # normally run
            # distance_current_to_last = self._distance(env_robot_x, env_robot_y, self._env_robot_x_last, self._env_robot_y_last)
            # distance_current_to_goal = self._distance(env_robot_x, env_robot_y, env_goal_x, env_goal_y)
            # distance_last_to_goal = self._distance(self._env_robot_x_last, self._env_robot_y_last, env_goal_x, env_goal_y)
            # distance_difference_to_goal = abs(distance_current_to_goal - distance_last_to_goal)
            # # calculate the ratio of reward for moving to the goal
            # if distance_current_to_last != 0:
            #     reward_goal_ratio = distance_difference_to_goal / distance_current_to_last
            # else:
            #     reward_goal_ratio = 0
            # # calculate the reward for approaching to the goal
            # if distance_current_to_goal <= sqrt(distance_current_to_last**2+distance_last_to_goal**2):
            #     reward_goal = self._approach_goal_reward * reward_goal_ratio
            # else:
            #     reward_goal = - 2 * self._approach_goal_reward * reward_goal_ratio
            # # calculate the reward for arrive at the middle goal
            # if distance_current_to_goal < (GoConfig.STEPPING_VELOCITY * GoConfig.STEPPING_TIME / 2):
            #     reward_goal = self._reach_goal_reward
            # # ---------after---------------------
            # # normally run
            # # calculate the reward for approaching to the goal
            # # distance
            # distance_current_to_last = self._distance(env_robot_x, env_robot_y, self._env_robot_x_last, self._env_robot_y_last)
            # distance_current_to_goal = self._distance(env_robot_x, env_robot_y, env_goal_x, env_goal_y)
            # distance_last_to_goal = self._distance(self._env_robot_x_last, self._env_robot_y_last, env_goal_x, env_goal_y)
            # distance_difference_to_goal = distance_current_to_goal - distance_last_to_goal
            #
            # if distance_current_to_last != 0:
            #     reward_goal_ratio = distance_difference_to_goal / distance_current_to_last
            #     if reward_goal_ratio > 1:
            #         reward_goal_ratio = 1
            #     if reward_goal_ratio < -1:
            #         reward_goal_ratio = -1
            # else:
            #     reward_goal_ratio = 0
            #
            # if distance_current_to_goal <= distance_last_to_goal:
            #     reward_goal = - self._approach_goal_reward * reward_goal_ratio
            # else:
            #     reward_goal = - self._approach_goal_reward * reward_goal_ratio
            #     # reward_goal = - self._approach_goal_reward
            # # print("reward_goal_ratio", reward_goal_ratio)
            # # print("reward_goal 1", reward_goal)
            # # angle
            # dir_robot_x = env_robot_x - self._env_robot_x_last
            # dir_robot_y = env_robot_y - self._env_robot_y_last
            # dir_route_x = env_goal_x - last_goal_x
            # dir_route_y = env_goal_y - last_goal_y
            # cos_fi = dir_robot_x * dir_route_x + dir_robot_y * dir_route_y
            # norm = math.sqrt((dir_robot_x ** 2 + dir_robot_y ** 2) * (dir_route_x ** 2 + dir_route_y ** 2))
            # if norm == 0:
            #     cos_fi = 0
            # else:
            #     cos_fi /= norm
            # reward_goal += self._approach_goal_reward * cos_fi
            # # print(last_goal_x, last_goal_y, env_goal_x, env_goal_y)
            # # print(dir_robot_x, dir_robot_y, dir_route_x, dir_route_y)
            # # print("cos_fi", cos_fi)
            # # print("reward_goal 2", reward_goal)
            # # calculate the reward for arrive at the middle goal
            # # if distance_current_to_goal < (GoConfig.STEPPING_VELOCITY * GoConfig.STEPPING_TIME / 2):
            # #     reward_goal = self._reach_goal_reward
            # # print("reward_goal 3", reward_goal)
            # # ---------third edition---------------------
            # normally run
            # calculate the reward for approaching to the goal according to the distance
            distance_current_to_last = self._distance(env_robot_x, env_robot_y, self._env_robot_x_last, self._env_robot_y_last)
            distance_current_to_goal = self._distance(env_robot_x, env_robot_y, env_goal_x, env_goal_y)
            distance_last_to_goal = self._distance(self._env_robot_x_last, self._env_robot_y_last, env_goal_x, env_goal_y)
            distance_difference_to_goal = distance_current_to_goal - distance_last_to_goal
            if distance_current_to_last != 0:
                if distance_difference_to_goal > 0:
                    reward_goal_ratio = -1
                else:
                    reward_goal_ratio = 0
            else:
                reward_goal_ratio = 0
            reward_goal = self._approach_goal_reward * reward_goal_ratio
            # mean velocity penalty
            if v_mean == 0:
                reward_goal = - 5
            # reach goal reward
            if distance_current_to_goal < (GoConfig.STEPPING_VELOCITY * GoConfig.STEPPING_TIME / 4):
                reward_goal = self._reach_goal_reward
        elif env_done == 3:
            # reach the goal
            reward_goal = self._reach_goal_reward
        else:
            reward_goal = 0

        # reward for other agvs
        if env_done == 0:
            # normally run
            if (nearest_other_agent_dis >= self._other_agv_distance_threshold) and \
                    (nearest_other_agent_dis <= self._max_factor * self._other_agv_distance_threshold):
                reward_other_agvs = 0.25 * self._collision_penalty * (self._max_factor * self._other_agv_distance_threshold
                                                               - nearest_other_agent_dis) / self._other_agv_distance_threshold
            else:
                reward_other_agvs = 0
        elif env_done == 1:
            # collision
            reward_other_agvs = self._collision_penalty
            done = True
        else:
            reward_other_agvs = 0

        # reward for allow area
        if env_done == 0:
            # normally run
            # if (nearest_allow_area_dis > self._allow_area_distance_threshold) and \
            #         (nearest_allow_area_dis < self._max_factor * self._allow_area_distance_threshold):
            #     reward_allow_area = self._collision_penalty * (self._max_factor * self._allow_area_distance_threshold
            #                                                    - nearest_allow_area_dis) / self._allow_area_distance_threshold
            # else:
            #     reward_allow_area = 0
            reward_allow_area = 0
        elif env_done == 2:
            # collision
            reward_allow_area = self._collision_penalty
            done = True
        else:
            reward_allow_area = 0

        # reward for road network
        if env_done == 0:
            # normally run
            if nearest_road_dis >= self._road_network_distance_threshold:
                reward_road_network = self._deviate_road_penalty
            elif (nearest_road_dis > self._min_factor * self._road_network_distance_threshold) and \
                    (nearest_road_dis < self._road_network_distance_threshold):
                reward_road_network = self._collision_penalty * (nearest_road_dis - self._min_factor * self._road_network_distance_threshold) \
                                      / self._road_network_distance_threshold
            else:
                reward_road_network = 0
        else:
            reward_road_network = 0

        reward = reward_goal + reward_other_agvs + reward_allow_area + reward_road_network

        self._env_robot_x_last = env_robot_x
        self._env_robot_y_last = env_robot_y
        self._env_robot_orientation_last = env_robot_orientation

        return reward, done

    def calculate_optimal_path_reward(self, env_done, shortest_path_dis, nearest_road_dis, nearest_allow_area_dis,
                                       nearest_other_agent_dis, agents_average_velocity, block):
        """
        Calculate the reward for one step
        :param env_robot_x:
        :param env_robot_y:
        :param env_robot_orientation:
        :param env_goal_x:
        :param env_goal_y:
        :param env_done:
        :param nearest_road_dis:
        :param nearest_allow_area_dis:
        :param nearest_other_agent_dis:
        :return:
        Return reward and done
        """
        # reward = 1
        # done = 1

        done = False

        # reward for goal
        if env_done == 0:
            # normally run
            # calculate the reward for approaching to the goal
            if self.last_optimal_path_len > shortest_path_dis:
                reward_goal = 0.5
            else:
                reward_goal = -1
            # average velocity
            if agents_average_velocity >= self._last_agents_average_velocity:
                reward_goal += 0
            else:
                reward_goal += 0
            # block penalty
            if block:
                reward_goal += -1
        elif env_done == 3:
            # reach the goal
            reward_goal = self._reach_goal_reward
        else:
            reward_goal = 0

        # reward for other agvs
        if env_done == 0:
            # normally run
            if (nearest_other_agent_dis >= self._other_agv_distance_threshold) and \
                    (nearest_other_agent_dis <= self._max_factor * self._other_agv_distance_threshold):
                reward_other_agvs = 0.25 * self._collision_penalty * (self._max_factor * self._other_agv_distance_threshold
                                                               - nearest_other_agent_dis) / self._other_agv_distance_threshold
            else:
                reward_other_agvs = 0
        elif env_done == 1:
            # collision
            reward_other_agvs = self._collision_penalty
            done = True
        else:
            reward_other_agvs = 0

        # reward for allow area
        if env_done == 0:
            # normally run
            # if (nearest_allow_area_dis > self._allow_area_distance_threshold) and \
            #         (nearest_allow_area_dis < self._max_factor * self._allow_area_distance_threshold):
            #     reward_allow_area = self._collision_penalty * (self._max_factor * self._allow_area_distance_threshold
            #                                                    - nearest_allow_area_dis) / self._allow_area_distance_threshold
            # else:
            #     reward_allow_area = 0
            reward_allow_area = 0
        elif env_done == 2:
            # collision
            reward_allow_area = self._collision_penalty
            done = True
        else:
            reward_allow_area = 0

        # reward for road network
        if env_done == 0:
            # normally run
            if nearest_road_dis >= self._road_network_distance_threshold:
                reward_road_network = self._deviate_road_penalty
            elif (nearest_road_dis > self._min_factor * self._road_network_distance_threshold) and \
                    (nearest_road_dis < self._road_network_distance_threshold):
                reward_road_network = self._collision_penalty * (nearest_road_dis - self._min_factor * self._road_network_distance_threshold) \
                                      / self._road_network_distance_threshold
            else:
                reward_road_network = 0
        else:
            reward_road_network = 0

        reward = reward_goal + reward_other_agvs + reward_allow_area + reward_road_network

        self.last_optimal_path_len = shortest_path_dis
        self._last_agents_average_velocity = agents_average_velocity

        return reward, done

    def calculate_road_map_reward(self, env_done, nearest_other_agent_dis, deadlock, velocity):
        """
        Calculate the reward for the road map environment
        Return reward and done
        """
        # reward = 1
        # done = 1

        done = False

        # reward for goal
        if env_done == 0 or env_done == 5:
            # normally run
            # calculate the reward for approaching to the goal
            if velocity > 0:
                reward_goal = 0
            else:
                reward_goal = 0
        elif env_done == 3:
            # reach the goal
            reward_goal = self._reach_goal_reward
        else:
            reward_goal = 0

        # reward for other agvs
        if env_done == 0 or env_done == 5:
            # normally run
            if (nearest_other_agent_dis >= self._other_agv_distance_threshold) and \
                    (nearest_other_agent_dis <= self._max_factor * self._other_agv_distance_threshold):
                reward_other_agvs = 0 * self._collision_penalty * (self._max_factor * self._other_agv_distance_threshold
                                                               - nearest_other_agent_dis) / self._other_agv_distance_threshold
            else:
                reward_other_agvs = 0
        elif env_done == 1:
            # collision
            reward_other_agvs = self._collision_penalty
            done = True
        else:
            reward_other_agvs = 0

        if deadlock:
            reward_deadlock = self._collision_penalty
            done = True
        else:
            reward_deadlock = 0

        reward = reward_goal + reward_other_agvs + reward_deadlock

        return reward, done

    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate the euler distance from to point.
        :param x1: First point x.
        :param y1: First point y.
        :param x2: Second point x.
        :param y2: Second point y.
        :return: Euler distnace from to points.
        """
        x = x1 - x2
        y = y1 - y2
        return math.sqrt(x * x + y * y)
