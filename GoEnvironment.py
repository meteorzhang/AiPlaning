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


import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc
import sys
import math
from GoConfig import GoConfig
from GoEmulator import GoEmulator
# from go_ga3c.GoStatus import FullStatus
# from environment.environment import Environment


class GoEnvironment:
    def __init__(self, id=-1, shared_env=None, actions=None, enable_show=True):
        self.id = id
        # add environment emulator
        self.shared_env = shared_env
        self.nb_frames = GoConfig.STACKED_FRAMES
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        self.visualize = enable_show
        self.actions = actions
        self.step_time = GoConfig.STEPPING_TIME
        self.env = GoEmulator(self.id, self.shared_env, self.visualize)
        # self.reset()

    # 将数据转化成数组.curent status include static status and dynamic status.
    def get_full_current_state(self):
        # x_ = self.shared_env.get_full_status(self.id)
        # x_, other_ = self.shared_env.get_local_full_status(self.id)
        # x_, other_ = self.shared_env.get_local_full_polar_status(self.id)
        x_, other_ = self.shared_env.get_global_status(self.id)
        return x_, other_

    def get_self_current_state(self):
        return self.shared_env.get_self_current_status(self.id)

    def _get_other_agent_state(self):
        return self.shared_env.get_other_agents_status(self.id)

    def reset(self):
        self.total_reward = 0
        status = self.env.reset()
        self.shared_env.update_agent_status(self.id, status)
        self.previous_state = self.current_state = status
        # self.shared_environment.update_agent_status(self.id,self.current_state)
        # self.frame_q.queue.clear()
        # self._update_frame_q(self.game.reset())

    def step_env(self, action, other_state):
        self._update_display(other_state)
        if action is None:
            observation, reward, done, forced_stop = self.env.step(0, other_state, self.step_time)
            reward = 0
            done = False
        else:
            v = self.actions.get_action_by_indx(action)
            observation, reward, done, forced_stop = self.env.step(v, other_state, self.step_time)
        return observation, reward, done, forced_stop

    def step(self, action, other_agent_state):
        observation, reward, done, forced_stop = self.step_env(action, other_agent_state)
        self.previous_state = self.current_state
        self.shared_env.update_agent_status(observation.id, observation)
        self.total_reward += reward
        self.current_state, _ = self.get_full_current_state()
        return reward, done, forced_stop

    def _update_display(self, other_agent_state):
        # if self.visualize:
        #    self.env.visualize(other_agent_state)
        pass

    def observation_size(self):
        pass
        # return self.env.observation_size()

    def close(self):
        self.env.close()
