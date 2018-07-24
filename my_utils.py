# coding=utf-8
import numpy as np
from model.config import cfg


def random_range(max_value):
    """
    :param max_value: the max value is max_value - 1
    :return: int
    """
    return int(np.random.random() * max_value)

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    print('indices: {} - values: {} - shape: {}'.format(indices, values, shape))

    return indices, values, shape

def transform_for_ctc(label):
    """
    TODO 是否有必要？
    生成供ctc训练的label
    """
    np.asarray([cfg.MY.SPACE_INDEX if x == cfg.MY.SPACE_TOKEN else () for x in label])