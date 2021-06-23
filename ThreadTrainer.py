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

from threading import Thread
import numpy as np
from PIL import Image
import time
from go_ga3c.GoConfig import GoConfig


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False
        self.batch_size = 0

    def run(self):
        # List for training
        # # maps = []
        # selfs = []
        # others = []
        # observations = []
        # other_num = []
        # rewards = []
        # actions = []
        # state_ = []
        # while not self.exit_flag:
        #     if self.server.training_q.empty():
        #         time.sleep(0.001)
        #         continue
        #     state, r_, a_ = self.server.training_q.get()
        #     # print('Thread Traning len(stat)=%d len(r_)=%d len(a_)=%d' % (len(state), len(r_), len(a_)))
        #     if GoConfig.TRAIN_MODELS:
        #         for i in range(len(state)):
        #             state_.append(state[i])
        #             np_state = state[i].to_numpy_array()
        #             # maps.append(np_state['map'].copy())
        #             selfs.append(np_state['self'].copy())
        #             observations.append(np_state['observation'].copy())
        #             other_agent_state = np_state['other'].copy()
        #             other_static = np.zeros((GoConfig.A_MAP_MAX_AGENT_SIZE - 1, GoConfig.LOCAL_OTHER_STATUS_SIZE), dtype=np.float)
        #             o_num = len(other_agent_state)
        #             for idx in range(o_num):
        #                 other_static[idx] = np.array(other_agent_state[idx])
        #             others.append(other_static)
        #             other_num.append(o_num)
        #             rewards.append(r_[i])
        #             # print("reward", rewards)
        #             actions.append(a_[i])
        #             if len(selfs) >= GoConfig.TRAINING_MIN_BATCH_SIZE:
        #                 action_b = np.array(actions).reshape((len(actions), self.server.process_agent_manager.actions.action_space_nums))
        #                 reward_b = np.array(rewards).reshape(len(rewards))
        #                 # print(rewards.shape)
        #                 # map_image1 = Image.fromarray(maps[0].astype('uint8'))
        #                 # map_image1.show()
        #                 # print(observations[].shape)
        #                 self.server.train_model(selfs, others, observations, other_num, reward_b, action_b)
        #                 selfs[:] = []
        #                 observations[:] = []
        #                 others[:] = []
        #                 other_num[:] = []
        #                 rewards[:] = []
        #                 actions[:] = []
        #                 state_[:] = []
        #                 # for i in range(len(state_)):
        #                     # state_[i].clear()

        # numpy for training
        while not self.exit_flag:
            if self.server.training_q.empty():
                time.sleep(0.001)
                continue

            state, r, a = self.server.training_q.get()

            if GoConfig.TRAIN_MODELS:
                for index, item in enumerate(state):
                    np_state = item.to_numpy_array()
                    if self.batch_size == 0:
                        selfs = np_state['self'][np.newaxis, :]
                        if np_state['other'].shape[0] == 0:
                            others = np.zeros((GoConfig.A_MAP_MAX_AGENT_SIZE - 1, GoConfig.LOCAL_OTHER_STATUS_SIZE),
                                              dtype=np.float)[np.newaxis, :]
                        else:
                            others = np.vstack((np_state['other'], np.zeros(
                                (GoConfig.A_MAP_MAX_AGENT_SIZE - 1 - np_state['other'].shape[0],
                                 GoConfig.LOCAL_OTHER_STATUS_SIZE), dtype=np.float)))[np.newaxis, :]
                        other_num = np.array([np_state['other'].shape[0]])
                        path_length = np.array([np_state['path'].shape[0]])
                        paths = np.vstack((np_state['path'],
                                           np.zeros((GoConfig.MAX_PATH_LENGTH - np_state['path'].shape[0], 2),
                                                    dtype=np.float)))[np.newaxis, :]
                        rewards = np.array([r[index]])
                        actions = a[index][np.newaxis, :]
                    else:
                        selfs = np.concatenate((selfs, np_state['self'][np.newaxis, :]))
                        if np_state['other'].shape[0] == 0:
                            other = np.zeros((GoConfig.A_MAP_MAX_AGENT_SIZE - 1, GoConfig.LOCAL_OTHER_STATUS_SIZE),
                                             dtype=np.float)[np.newaxis, :]
                        else:
                            other = np.vstack((np_state['other'], np.zeros(
                                (GoConfig.A_MAP_MAX_AGENT_SIZE - 1 - np_state['other'].shape[0],
                                 GoConfig.LOCAL_OTHER_STATUS_SIZE), dtype=np.float)))[np.newaxis, :]
                        others = np.concatenate((others, other))
                        other_num = np.concatenate((other_num, np.array([np_state['other'].shape[0]])))
                        path_length = np.concatenate((path_length, np.array([np_state['path'].shape[0]])))
                        path = np.vstack((np_state['path'],
                                          np.zeros((GoConfig.MAX_PATH_LENGTH - np_state['path'].shape[0], 2), dtype=np.float)))[np.newaxis, :]
                        paths = np.concatenate((paths, path))
                        rewards = np.concatenate((rewards, np.array([r[index]])))
                        actions = np.concatenate((actions, a[index][np.newaxis, :]))
                    self.batch_size += 1

                    if self.batch_size >= GoConfig.TRAINING_MIN_BATCH_SIZE:
                        self.batch_size = 0
                        self.server.train_model(selfs, others, other_num, paths, path_length, rewards, actions)
