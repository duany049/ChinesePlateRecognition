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
        # indices.extend(zip([n] * len(seq), range(len(seq))))
        # 测试项目,序列只有1个 TODO 改成车牌时候的时候要改过来
        indices.extend(zip([n] * 1, range(1)))
        seq = [seq]
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    # print('indices: {} - values: {} - shape: {}'.format(indices, values, shape))

    return indices, values, shape

# def decode_a_seq(indexes, spars_tensor):
#     str_decoded = ''.join([common.CHARS[spars_tensor[1][m] - common.FIRST_INDEX] for m in indexes])
#     # Replacing blank label to none
#     str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
#     # Replacing space label to space
#     str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
#     # print("ffffffff", str_decoded)
#     return str_decoded

def decode_sparse_tensor(sparse_tensor):
    # print(sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    #
    # print("mmmm", decoded_indexes)
    for index in decoded_indexes:
        str_decoded = [sparse_tensor[1][m] for m in index]
        print('str_decoded: {}'.format(str_decoded))
    # result = []
    # for index in decoded_indexes:
    #     result.append(decode_a_seq(index, sparse_tensor))
    # return result

def transform_for_ctc(label):
    """
    TODO 是否有必要？
    生成供ctc训练的label
    """
    np.asarray([cfg.MY.SPACE_INDEX if x == cfg.MY.SPACE_TOKEN else () for x in label])