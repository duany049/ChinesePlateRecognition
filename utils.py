# coding=utf-8
import numpy as np


def random_range(max_value):
    """
    :param max_value: the max value is max_value - 1
    :return: int
    """
    return int(np.random.random() * max_value)
