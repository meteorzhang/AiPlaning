from collections import namedtuple
import math

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])
ActionVTh = namedtuple('ActionVTh', ['v', 'theta'])
ActionGlobal = namedtuple('ActionGlobal', ['v'])


def map_action(action, stepping_velocity):
    if action == 0:
        vx = 0
        vy = stepping_velocity
    elif action == 1:
        vx = 0
        vy = - stepping_velocity
    elif action == 2:
        vx = stepping_velocity
        vy = 0
    elif action == 3:
        vx = - stepping_velocity
        vy = 0
    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return vx, vy


def global_action(action, stepping_velocity):
    if action == 0:
        v = 0
    elif action == 1:
        v = stepping_velocity
    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return v


def polar_action(action, stepping_velocity):
    """
    agents action in polar coordinate:
    4   3   2
    5   0   1
    6   7   8

        2
    3   0   1
        4

    Parameters
    ----------
    action: action index
    stepping_velocity:

    Returns
    -------
    v:
    theta:

    Raises
    ------
    Invalid action
    """
    """
    if action == 0:
        v = 0
        theta = math.pi * 2
    elif action == 1:
        v = stepping_velocity
        theta = math.pi * 2
    elif action == 2:
        v = stepping_velocity
        theta = math.pi / 4
    elif action == 3:
        v = stepping_velocity
        theta = math.pi / 2
    elif action == 4:
        v = stepping_velocity
        theta = math.pi * 3 / 4
    elif action == 5:
        v = stepping_velocity
        theta = math.pi
    elif action == 6:
        v = stepping_velocity
        theta = math.pi * 5 / 4
    elif action == 7:
        v = stepping_velocity
        theta = math.pi * 3 / 2
    elif action == 8:
        v = stepping_velocity
        theta = math.pi * 7 / 4
    else:
        raise AttributeError("Invalid Action: {}".format(action))
    """
    if action == 0:
        v = 0
        theta = math.pi * 2
    elif action == 1:
        v = stepping_velocity
        theta = math.pi * 2
    elif action == 2:
        v = stepping_velocity
        theta = math.pi / 2
    elif action == 3:
        v = stepping_velocity
        theta = math.pi
    elif action == 4:
        v = stepping_velocity
        theta = math.pi * 3 / 2
    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return v, theta
