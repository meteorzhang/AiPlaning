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

from go_ga3c.GoConfig import GoConfig
import time


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            if self.server.prediction_q.empty():
                time.sleep(0.001)
                continue

            size = 0

            while size < GoConfig.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                id, state = self.server.prediction_q.get()
                np_state = state.to_numpy_array()
                if size == 0:
                    ids = np.array([id])
                    selfs = np_state['self'][np.newaxis, :]

                    path_length = np.array([np_state['path'].shape[0]])
                    paths = np.vstack((np_state['path'],
                                      np.zeros((GoConfig.MAX_PATH_LENGTH - np_state['path'].shape[0], 2), dtype=np.float)))[np.newaxis, :]

                    other_num = np.array([np_state['other'].shape[0]])
                    if np_state['other'].shape[0] == 0:
                        others = np.zeros((GoConfig.A_MAP_MAX_AGENT_SIZE - 1, GoConfig.LOCAL_OTHER_STATUS_SIZE),
                                          dtype=np.float)[np.newaxis, :]
                    else:
                        others = np.vstack((np_state['other'], np.zeros(
                            (GoConfig.A_MAP_MAX_AGENT_SIZE - 1 - np_state['other'].shape[0],
                             GoConfig.LOCAL_OTHER_STATUS_SIZE), dtype=np.float)))[np.newaxis, :]
                else:
                    ids = np.concatenate((ids, np.array([id])))
                    selfs = np.concatenate((selfs, np_state['self'][np.newaxis, :]))

                    path_length = np.concatenate((path_length, np.array([np_state['path'].shape[0]])))
                    path = np.vstack((np_state['path'],
                                      np.zeros((GoConfig.MAX_PATH_LENGTH - np_state['path'].shape[0], 2), dtype=np.float)))[np.newaxis, :]
                    paths = np.concatenate((paths, path))

                    other_num = np.concatenate((other_num, np.array([np_state['other'].shape[0]])))
                    if np_state['other'].shape[0] == 0:
                        other = np.zeros((GoConfig.A_MAP_MAX_AGENT_SIZE - 1, GoConfig.LOCAL_OTHER_STATUS_SIZE),
                                         dtype=np.float)[np.newaxis, :]
                    else:
                        other = np.vstack((np_state['other'], np.zeros(
                            (GoConfig.A_MAP_MAX_AGENT_SIZE - 1 - np_state['other'].shape[0],
                             GoConfig.LOCAL_OTHER_STATUS_SIZE), dtype=np.float)))[np.newaxis, :]
                    others = np.concatenate((others, other))
                size += 1

            if size > 0:
                p, v = self.server.model.predict_p_and_v(selfs, others, other_num, paths, path_length)

            for i in range(size):
                agent = self.server.process_agent_manager.get_agent_by_id(ids[i])
                if agent is not None:
                    agent.wait_q.put((p[i], v[i]))
