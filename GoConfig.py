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

# from environment.environment_fitness import Mode


class GoConfig:

    #########################################################################
    # Environment configuration

    # Mode
    # MODE=Mode.ALL_RANDOM
    # Terminate the simulation
    TERMINATE_AT_END = False
    # Cluster size of the lidar
    CLUSTER_SIZE = 1
    # use observation rotation vector
    USE_OBSERVATION_ROTATION = True
    # Observation rotation vector size
    OBSERVATION_ROTATION_SIZE = 128

    # ----------------------AGENT RELATED-------------------------- #
    # ACTION
    MAX_SPEED = 1
    SPEED_SAMPLES = 5
    ROTATION_SAMPLES = 3

    STEPPING_VELOCITY = 2
    STEPPING_TIME = 1

    ACTION_SIZE = 2

    # Status:
    STATUS_SIZE = 14
    LOCAL_SELF_STATUS_SIZE = 12
    LOCAL_OTHER_STATUS_SIZE = 13
    A_MAP_MAX_AGENT_SIZE = 5
    MAX_PATH_LENGTH = 5

    # Agent config
    AGENT_CONFIG_FILE = '../agent_config_demo.ini'
    RADIUS = 0.5    # 0.2 radius, 0.05 safe distance
    TARGET_RANGE = 50
    OBSERVATION_RANGE = 10
    OBSERVATION_ANGLE = 0.5  # angle
    OBSERVATION_ANGLE_SIZE = 720
    OBSERVATION_ANGLE_CHANNEL = 1
    ALLOW_ACTION = 5

    # ---------------------TRAIN RELATED--------------------------- #
    # IMAGE INPUT INTO THE NET
    WIDTH = 84
    HEIGHT = 84
    CHANNEL = 3
    MAP_RESOLUTION = 0.05  # resolution: ????????????

    # Visualize for training
    VISUALIZE = True
    # Enable to see the trained agent in action
    PLAY_MODE = False
    # Enable to train
    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = True
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 0

    #########################################################################
    # Number of agents, predictors, trainers and other system settings

    # If the dynamic configuration is on, these are the initial values.
    # Number of Agents
    AGENTS = 64  # 32
    # Number of Predictors
    PREDICTORS = 8  # 2
    # Number of Trainers
    TRAINERS = 8  # 2

    # Device
    DEVICE = ['/gpu:0']
    # DEVICE = '/gpu:0'


    # Enable the dynamic adjustment (+ waiting time to start it)
    DYNAMIC_SETTINGS = False
    DYNAMIC_SETTINGS_STEP_WAIT = 20
    DYNAMIC_SETTINGS_INITIAL_WAIT = 10

    #########################################################################
    # Algorithm parameters

    # Max step Iteration -> if read the environment ist done. 0 for endless.
    MAX_STEP_ITERATION = 300

    # Discount factor
    DISCOUNT = 0.99

    # Tmax
    TIME_MAX = 10 #5

    # Reward Clipping
    REWARD_MIN = -15
    REWARD_MAX = 12

    # Max size of the queue
    MAX_QUEUE_SIZE = 800 #100
    PREDICTION_BATCH_SIZE = 128 #128

    # Input of the DNN
    STACKED_FRAMES = 4
    OBSERVATION_SIZE = 1081 + OBSERVATION_ROTATION_SIZE

    # Total number of episodes and annealing frequency, learning rate
    # Asynchronous training episodes
    # EPISODES = 50008
    # ANNEALING_EPISODE_COUNT = EPISODES
    # LEARNING_RATE_START = 0.0003
    # LEARNING_RATE_END = 0.0001
    # Synchronous training episodes
    EPISODES = 400012
    ANNEALING_EPISODE_COUNT = EPISODES * 3 / 4
    LEARNING_RATE_START = 0.0003
    LEARNING_RATE_END = 0.000003

    # Entropy regualrization hyper-parameter
    BETA_START = 0.01
    BETA_END = 0.01

    TRAIN_BATCH_SIZE = 4

    # RMSProp parameters
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1

    # Dual RMSProp
    DUAL_RMSPROP = True

    # Gradient clipping
    USE_GRAD_CLIP = False
    GRAD_CLIP_NORM = 40.0
    # Epsilon (regularize policy lag in GA3C)
    LOG_EPSILON = 1e-6
    # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower
    # TRAINING_MIN_BATCH_SIZE = GPU_MIN_BATCH_SIZE * len(DEVICE)
    TRAINING_MIN_BATCH_SIZE = 16 # 0
    GPU_MIN_BATCH_SIZE = 16

    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = True
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 1000

    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 1
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 1000

    # Results filename
    RESULTS_FILENAME = 'results.txt'
    # Network checkpoint name
    NETWORK_NAME = 'network'

    #########################################################################
    # More experimental parameters here

    # Minimum policy
    MIN_POLICY = 0.0
    # Use log_softmax() instead of log(softmax())
    USE_LOG_SOFTMAX = False

    # Reward
    COLLISION_PENALTY = -20
    REACH_GOAL_REWARD = 20
    APPROACH_GOAL_REWARD = 5
    DEVIATE_ROAD_PENALTY = -20
    OTHER_AGV_DISTANCE_THRESHOLD = RADIUS * 2
    ALLOW_AREA_DISTANCE_THRESHOLD = RADIUS * 0
    ROAD_NETWORK_DISTANCE_THRESHOLD = RADIUS * 5
    MIN_FACTOR = 0.5
    MAX_FACTOR = 2.0
    # collision reward

    # pre-trained model
    LOAD_PRETRAIN_CHECKPOINT = False
    RESNET_V2_50_PRETRAIN_MODEL = "./Resnet_v2_50/resnet_v2_50.ckpt"
