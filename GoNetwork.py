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

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.slim import nets
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn

from go_ga3c.GoConfig import GoConfig
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
1.修改NN结构，包括对动态环境变化能力的感知，对精通地图、路网、通行区域的感知
2.采用Lstm+Cnn+GA3C,
3.重点
'''


class GoNetwork:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions
        self.status_size = GoConfig.STATUS_SIZE
        self.local_other_status_size = GoConfig.LOCAL_OTHER_STATUS_SIZE
        self.local_self_status_size = GoConfig.LOCAL_SELF_STATUS_SIZE
        self.other_agent_size = GoConfig.A_MAP_MAX_AGENT_SIZE-1
        self.maps_width = GoConfig.WIDTH
        self.maps_height = GoConfig.HEIGHT
        self.maps_channel = GoConfig.CHANNEL
        self.observation_size = GoConfig.OBSERVATION_SIZE
        self.observation_channels = GoConfig.STACKED_FRAMES
        self.learning_rate = GoConfig.LEARNING_RATE_START
        self.beta = GoConfig.BETA_START
        self.log_epsilon = GoConfig.LOG_EPSILON
        self.gpu_mini_batch_size = GoConfig.GPU_MIN_BATCH_SIZE
        self.pretrain_model_path = GoConfig.RESNET_V2_50_PRETRAIN_MODEL
        self.observation_angle_size = GoConfig.OBSERVATION_ANGLE_SIZE
        self.observation_allow_action = GoConfig.ALLOW_ACTION
        self.variables_to_restore = []
        self.max_path_length = GoConfig.MAX_PATH_LENGTH

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            # Graph input
            self.x = tf.placeholder(tf.float32, [None, self.local_self_status_size], name='self_status')
            self.x_other = tf.placeholder(tf.float32, [None, self.other_agent_size, self.local_other_status_size],
                                          name='other_status')
            self.paths = tf.placeholder(tf.float32, [None, self.max_path_length, 2], name='paths')
            # self.maps = tf.placeholder(tf.float32, [None, self.maps_width, self.maps_height, self.maps_channel],
            #                            name='static_map')
            self.y_r = tf.placeholder(tf.float32, [None], name='reward')
            self.action_index = tf.placeholder(tf.float32, [None, self.num_actions], name='action_idx')
            # Graph constant
            self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
            self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.other_num = tf.placeholder(tf.int32)
            self.path_length = tf.placeholder(tf.int32)
            # Calculate the gradients for each model tower
            self.tower_v_grads = []
            self.tower_p_grads = []
            self.tower_all_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                # Loop for all GPUs
                for i in range(len(self.device)):
                    with tf.device(self.device[i]):
                        with tf.name_scope("tower_%d" % i):
                            # Split data between GPUs
                            _x = self.x[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            _x_other = self.x_other[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            _paths = self.paths[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            # _maps = self.maps[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            _y_r = self.y_r[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            _action_index = self.action_index[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            _other_num = self.other_num[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            _path_length = self.path_length[i*self.gpu_mini_batch_size:(i+1)*self.gpu_mini_batch_size]
                            # Training
                            self._deep_neural_network(_x, _x_other, _other_num, _paths, _path_length)
                            # Reuse variables
                            tf.get_variable_scope().reuse_variables()
                            # Cost
                            self._cost_function(_y_r, _action_index)
                            # Optimizer
                            self._rmspo_optimizer()
                            # Gradient
                            self._compute_current_gradients()
                            # Prediction
                            if i == 0:
                                self._deep_neural_network(self.x, self.x_other, self.other_num, self.paths, self.path_length)
                                self._cost_function(self.y_r, self.action_index)
                                self.logits_v_predictor = self.logits_v
                                self.softmax_p_predictor = self.softmax_p
            # Average gradient
            self._compute_average_gradients()
            # Apply average gradient
            self._apply_average_gradients()

            self.sess = tf.Session(
                graph=self.graph,
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.99)))
            self.sess.run(tf.global_variables_initializer())

            if GoConfig.TENSORBOARD: self._create_tensor_board()
            if GoConfig.LOAD_CHECKPOINT or GoConfig.SAVE_MODELS:
                vars = tf.global_variables()
                self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=10)
            if GoConfig.LOAD_PRETRAIN_CHECKPOINT:
                self.variables_to_restore = self.get_restore_variables('OtherStateNet, InputNet, PathNet'
                                                                       'Logits, FusionNet, OutputNet, global_step')
                # for var in self.variables_to_restore:
                self.get_init_fn(self.variables_to_restore)
                # self.variables_to_train = self.get_trainable_variables('resnet_v2_50')

    def _deep_neural_network(self, x, x_other, other_num, paths, path_length):
        with tf.variable_scope('OtherStateNet'):
            # # lstm对其他车辆编码
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128, name="LSTM_CELL")
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_other, dtype=tf.float32, sequence_length=other_num)
            other_s = states.h
            other_s_dense = self.dense_layer(other_s, 512, 'other_s_dense')

        with tf.variable_scope('PathNet'):
            # # lstm对其他车辆编码
            path_cell = tf.nn.rnn_cell.LSTMCell(num_units=64, name="PATH_CELL")
            outputs1, states1 = tf.nn.dynamic_rnn(path_cell, paths, dtype=tf.float32, sequence_length=path_length)
            path_states = states1.h
            path_states_dense = self.dense_layer(path_states, 256, 'path_states_dense')

        with tf.variable_scope('InputNet'):
            # _x_shape = x.get_shape()
            # nb_elements = _x_shape[1] * _x_shape[2]
            # self.x_flat = tf.reshape(x, shape=[-1, nb_elements._value])
            x_dense = self.dense_layer(x, 256, 'x_dense')

        with tf.variable_scope('FusionNet'):
            concat1 = tf.concat([x_dense, other_s_dense, path_states_dense], 1)
            # concat1 = tf.concat([x, x_observation, other_s], 1)
            # concat1 = tf.concat([x_dense1, observation_net_dense1], 1)
            # concat1 = tf.concat([x_dense1, observation_net_flat, other_s], 1)
            den1 = self.dense_layer(concat1, 2048, 'dens1')
            den2 = self.dense_layer(den1, 2048, 'dens2')

        with tf.variable_scope('OutputNet'):
            self.logits_v = tf.squeeze(self.dense_layer(den2, 1, 'logits_v', func=None), axis=[1])
            self.logits_p = self.dense_layer(den2, self.num_actions, 'logits_p', func=None)

    def _cost_function(self, y_r, a_index):
        # Cost of value
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(y_r - self.logits_v), axis=0)
        # Cost of policy
        if GoConfig.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * a_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                            tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + GoConfig.MIN_POLICY) / (
                        1.0 + GoConfig.MIN_POLICY * self.num_actions)
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * a_index, axis=1)

            self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                            * (y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                            tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                          self.softmax_p, axis=1)

        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)
        # Cost of value and policy
        self.cost_all = self.cost_p + self.cost_v

    def _rmspo_optimizer(self):
        if GoConfig.DUAL_RMSPROP:
            self.opt_p = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=GoConfig.RMSPROP_DECAY,
                momentum=GoConfig.RMSPROP_MOMENTUM,
                epsilon=GoConfig.RMSPROP_EPSILON)

            self.opt_v = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=GoConfig.RMSPROP_DECAY,
                momentum=GoConfig.RMSPROP_MOMENTUM,
                epsilon=GoConfig.RMSPROP_EPSILON)
        else:
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=GoConfig.RMSPROP_DECAY,
                momentum=GoConfig.RMSPROP_MOMENTUM,
                epsilon=GoConfig.RMSPROP_EPSILON)

    def _compute_current_gradients(self):
        if GoConfig.USE_GRAD_CLIP:
            if GoConfig.DUAL_RMSPROP:
                # value
                self.variables_to_train_v = self.get_trainable_variables('resnet_v2_50, OutputNet/logits_p, global_step')
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v, var_list=self.variables_to_train_v)
                self.opt_grad_v_clipped = [(tf.clip_by_norm(g, GoConfig.GRAD_CLIP_NORM), v)
                                           for g, v in self.opt_grad_v if not g is None]
                self.tower_v_grads.append(self.opt_grad_v_clipped)
                # policy
                self.variables_to_train_p = self.get_trainable_variables('resnet_v2_50, OutputNet/logits_v, global_step')
                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p, var_list=self.variables_to_train_p)
                self.opt_grad_p_clipped = [(tf.clip_by_norm(g, GoConfig.GRAD_CLIP_NORM), v)
                                           for g, v in self.opt_grad_p if not g is None]
                self.tower_p_grads.append(self.opt_grad_p_clipped)
            else:
                # all: value + policy
                self.variables_to_train_all = self.get_trainable_variables('resnet_v2_50, global_step')
                self.opt_grad = self.opt.compute_gradients(self.cost_all, var_list=self.variables_to_train_all)
                self.opt_grad_clipped = [(tf.clip_by_average_norm(g, GoConfig.GRAD_CLIP_NORM), v) for g, v in
                                         self.opt_grad]
                self.tower_all_grads.append(self.opt_grad_clipped)
        else:
            if GoConfig.DUAL_RMSPROP:
                # value
                self.variables_to_train_v = self.get_trainable_variables('resnet_v2_50, OutputNet/logits_p, global_step')
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v, var_list=self.variables_to_train_v)
                self.tower_v_grads.append(self.opt_grad_v)
                # policy
                self.variables_to_train_p = self.get_trainable_variables('resnet_v2_50, OutputNet/logits_v, global_step')
                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p, var_list=self.variables_to_train_p)
                self.tower_p_grads.append(self.opt_grad_p)
            else:
                # all: value + policy
                self.variables_to_train_all = self.get_trainable_variables('resnet_v2_50, global_step')
                self.opt_grad = self.opt.compute_gradients(self.cost_all, var_list=self.variables_to_train_all)
                self.tower_all_grads.append(self.opt_grad)

    def _compute_average_gradients(self):
        """
        Computer the average gradient from all GPU
        :return:
        """
        if GoConfig.USE_GRAD_CLIP:
            if GoConfig.DUAL_RMSPROP:
                # value
                self.v_average_grads = self.average_gradients(self.tower_v_grads)
                # policy
                self.p_average_grads = self.average_gradients(self.tower_p_grads)
            else:
                # all
                self.all_average_grads = self.average_gradients(self.tower_all_grads)
        else:
            if GoConfig.DUAL_RMSPROP:
                # value
                self.v_average_grads = self.average_gradients(self.tower_v_grads)
                # policy
                self.p_average_grads = self.average_gradients(self.tower_p_grads)
            else:
                # all
                self.all_average_grads = self.average_gradients(self.tower_all_grads)

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            # print("grad and var", grad_and_vars)
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _apply_average_gradients(self):
        if GoConfig.USE_GRAD_CLIP:
            if GoConfig.DUAL_RMSPROP:
                self.train_op_v = self.opt_v.apply_gradients(self.v_average_grads, global_step=self.global_step)
                self.train_op_p = self.opt_p.apply_gradients(self.p_average_grads, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.apply_gradients(self.all_average_grads, global_step=self.global_step)
        else:
            if GoConfig.DUAL_RMSPROP:
                self.train_op_v = self.opt_v.apply_gradients(self.v_average_grads, global_step=self.global_step)
                self.train_op_p = self.opt_p.apply_gradients(self.p_average_grads, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.apply_gradients(self.all_average_grads, global_step=self.global_step)

    def _create_graph(self):
        return None

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", tf.reduce_mean(self.cost_p_1_agg)))
        summaries.append(tf.summary.scalar("Pcost_entropy", tf.reduce_mean(self.cost_p_2_agg)))
        summaries.append(tf.summary.scalar("Pcost", tf.reduce_mean(self.cost_p)))
        summaries.append(tf.summary.scalar("Vcost", tf.reduce_mean(self.cost_v)))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_v", self.logits_v_predictor))
        summaries.append(tf.summary.histogram("activation_p", self.softmax_p_predictor))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.elu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.elu, padding='SAME'):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding=padding) + b
            if func is not None:
                output = func(output)

        return output

    def conv1d_layer(self, input, filter_size, out_dim, name, stride, func=tf.nn.elu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv1d(input, w, stride=stride, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x, x_other, other_num):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x, self.x_other: x_other,
                                                             self.other_num: other_num})
        return prediction

    def predict_p(self, x, x_other, other_num):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x, self.x_other: x_other,
                                                              self.other_num: other_num})
        return prediction
    
    def predict_p_and_v(self, x, x_other, other_num, paths, path_length):
        return self.sess.run([self.softmax_p_predictor, self.logits_v_predictor],
                             feed_dict={self.x: x, self.x_other: x_other, self.other_num: other_num,
                                        self.paths: paths, self.path_length: path_length})
    
    def train(self, x, x_other, other_num, paths, path_length, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.x_other: x_other, self.other_num: other_num, self.paths: paths,
                          self.path_length: path_length, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, x_other, other_num, paths, path_length, y_r, a):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.x_other: x_other, self.other_num: other_num, self.paths: paths,
                          self.path_length: path_length, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        filenames=re.split('/|_|\.', filename)
        return int(filenames[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        # filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if GoConfig.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(GoConfig.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)

        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))

    def get_restore_variables(self, checkpoint_exclude_scopes):
        if checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        else:
            return None
        variables_to_restore = []
        for var in tf.global_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_restore.append(var)
        return variables_to_restore

    def get_init_fn(self, variables_to_restore):
        model_path = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if GoConfig.LOAD_EPISODE > 0:
            model_path = self._checkpoint_filename(GoConfig.LOAD_EPISODE)

        if self.pretrain_model_path and (not model_path):
            if tf.gfile.IsDirectory(self.pretrain_model_path):
                checkpoint_path = tf.train.latest_checkpoint(self.pretrain_model_path)
            else:
                checkpoint_path = self.pretrain_model_path

            checkpoint_restore = tf.train.Saver(var_list=variables_to_restore)
            checkpoint_restore.restore(self.sess, checkpoint_path)

    def get_trainable_variables(self, exclude_scopes=None):
        if exclude_scopes:
            exclusions = [scope.strip() for scope in exclude_scopes.split(',')]
        else:
            return None
        variables_to_train = []
        for var in tf.global_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_train.append(var)
        return variables_to_train
