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

from multiprocessing import Queue
# from queue import Queue
import time
import sys

from GoConfig import GoConfig
from GoEnvironment import GoEnvironment
from GoNetwork import GoNetwork
from ProcessStats import ProcessStats
from ThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer
from GoProcessAgentManager import GoProcessAgentManger


class GoServer:
    def __init__(self):
        self.stats = ProcessStats()
        self.training_q = Queue(maxsize=GoConfig.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=GoConfig.MAX_QUEUE_SIZE)
        # agent管理类，完成对多个agent的运行状态的管理
        self.process_agent_manager = GoProcessAgentManger(self.prediction_q, self.training_q, self.stats.episode_log_q,
                                                          self.stats.episode_count)
        # network,神经网络核心类
        # self.model = GoNetwork(GoConfig.DEVICE, GoConfig.NETWORK_NAME, self.process_agent_manager.actions.action_space_nums)
        self.model = GoNetwork(GoConfig.DEVICE, GoConfig.NETWORK_NAME, self.process_agent_manager.actions.action_space_nums)
        if GoConfig.LOAD_CHECKPOINT:
            self.stats.episode_count.value = self.model.load()

        self.training_step = 0
        self.frame_counter = 0

        # self.agents = []
        self.predictors = []
        self.trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    #train(self, x,x_other,other_num,maps, y_r, a, trainer_id):
    def train_model(self, x, x_other, other_num, paths, path_length, y_r, a):
        self.model.train(x, x_other, other_num, paths, path_length, y_r, a)
        self.training_step += 1

        self.stats.training_count.value += 1
        self.dynamic_adjustment.temporal_training_count += 1

        if GoConfig.TENSORBOARD and self.stats.training_count.value % GoConfig.TENSORBOARD_UPDATE_FREQUENCY == 0:
            self.model.log(x, x_other, other_num, paths, path_length, y_r, a)

    def save_model(self):
        self.model.save(self.stats.episode_count.value)

    def main(self):
        # self.process_agent_manager.load_cfg()
        self.stats.start()
        self.dynamic_adjustment.start()

        if GoConfig.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        learning_rate_multiplier = (GoConfig.LEARNING_RATE_END - GoConfig.LEARNING_RATE_START) / GoConfig.ANNEALING_EPISODE_COUNT
        beta_multiplier = (GoConfig.BETA_END - GoConfig.BETA_START) / GoConfig.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < GoConfig.EPISODES:
            step = min(self.stats.episode_count.value, GoConfig.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = GoConfig.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = GoConfig.BETA_START + beta_multiplier * step

            if step > GoConfig.EPISODES / 2:
                self.process_agent_manager.epsilon = 0.95
            if step > GoConfig.ANNEALING_EPISODE_COUNT:
                self.process_agent_manager.epsilon = 0.99

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if GoConfig.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

            time.sleep(0.01)

        self.dynamic_adjustment.exit_flag = True
        while self.process_agent_manager.get_current_agent_num():
            self.process_agent_manager.terminate_agent()
        time.sleep(1)
        while self.predictors:
            self.remove_predictor()
        time.sleep(1)
        while self.trainers:
            self.remove_trainer()
