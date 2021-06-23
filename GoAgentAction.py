
from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
# import ActionXY,ActionRot
import itertools
import sys
import random
sys.path.append('../')
from go_ga3c.action_mapper import ActionXY, ActionRot, ActionVTh, ActionGlobal, map_action, polar_action, global_action
from go_ga3c.GoConfig import GoConfig


class GoAgentAction:
    # init actions,获得所有的actions的值
    def __init__(self):
        self.maxspeed = GoConfig.MAX_SPEED
        self.speed_samples = GoConfig.SPEED_SAMPLES
        self.stepping_velocity = GoConfig.STEPPING_VELOCITY
        self.rotation_samples = GoConfig.ROTATION_SAMPLES
        self.kinematics = 'none'
        self.action_space_nums = GoConfig.ACTION_SIZE
        self.action_space = None
        self.build_action_space(self.stepping_velocity)

    def get_all_actions(self):
        return self.action_space

    def get_actions_nums(self):
        return self.action_space_nums

    def get_action_by_indx(self, idx):
        # return self.action_space.index()
        return self.action_space[idx]

    def get_random_action(self):
        return random.choice(self.action_space)

    def build_action_space(self, stepping_velocity):
        """
        Action space

        """
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i+1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        print(speeds)
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 18, np.pi / 18, self.rotation_samples)
            print(rotations)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation)) 
                
        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space       
        """

        # Action in XY system
        # action_space = [ActionXY(0, 0)]
        #
        # for i in range(self.action_space_nums-1):
        #     vx, vy = map_action(i, stepping_velocity)
        #     action_space.append(ActionXY(vx, vy))

        # Action in polar system
        action_space = []
        for i in range(self.action_space_nums):
            v = global_action(i, stepping_velocity)
            action_space.append(v)

        self.action_space = action_space


if __name__ == '__main__':
    actions = GoAgentAction()

    for i in range(GoAgentAction().action_space_nums):
        print(i)
        print(actions.get_action_by_indx(i))
