import numpy as np


class AgentState(object):
    def __init__(self, px, py, vx, vy, last_local_gx, last_local_gy, next_local_gx, next_local_gy, global_gx, global_gy,
                 radius, v_mean, simple_paths, shortest_path_coordinate, mission_status, id):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.last_local_gx = last_local_gx
        self.last_local_gy = last_local_gy
        self.next_local_gx = next_local_gx
        self.next_local_gy = next_local_gy
        self.global_gx = global_gx
        self.global_gy = global_gy
        self.radius = radius
        self.v_mean = v_mean
        self.simple_paths = simple_paths
        self.shortest_path_coordinate = shortest_path_coordinate
        self.mission_status = mission_status
        self.id = id

    def update(self, px, py, vx, vy, last_local_gx, last_local_gy, next_local_gx, next_local_gy, global_gx, global_gy,
               radius, v_mean, simple_paths, shortest_path_coordinate, mission_status, id):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.last_local_gx = last_local_gx
        self.last_local_gy = last_local_gy
        self.next_local_gx = next_local_gx
        self.next_local_gy = next_local_gy
        self.global_gx = global_gx
        self.global_gy = global_gy
        self.radius = radius
        self.v_mean = v_mean
        self.simple_paths = simple_paths
        self.shortest_path_coordinate = shortest_path_coordinate
        self.mission_status = mission_status
        self.id = id

    def to_numpy_array(self):
        status = [self.px, self.py, self.vx, self.vy, self.last_local_gx, self.last_local_gy, self.next_local_gx,
                  self.next_local_gy, self.global_gx, self.global_gy, self.radius, self.v_mean, self.simple_paths,
                  self.shortest_path_coordinate, self.mission_status, self.id]
        return np.array(status)


class GlobalStatus:
    def __init__(self, self_status, other_status, path, status):
        self.self = self_status
        self.other = other_status
        self.path = path
        self.status = status

    def to_numpy_array(self):
        np_self = np.array(self.self)
        np_other = np.array(self.other)
        np_path = np.array(self.path)
        np_status = np.array(self.status)
        return {'self': np_self, 'other': np_other, 'path': np_path, 'status': np_status}


class LocalFullStatus:
    def __init__(self, self_observation, self_status, other_status):
        self.observation = self_observation
        self.self = self_status
        self.other = other_status

    def update(self, self_observation, self_status, other_status):
        self.observation = self_observation
        self.self = self_status
        self.other = other_status

    def to_numpy_array(self):
        np_observation = np.array(self.observation)
        np_self = np.array(self.self)
        np_other = np.array(self.other)
        return {'observation': np_observation, 'self': np_self, 'other': np_other}
