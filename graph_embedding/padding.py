# -*- coding: utf-8 -*-

import torch
import numpy as np


def expand_dim(tensor, dim, length):
    size = tensor.size()
    new_size = list(size[:dim])
    new_size.append(length)
    new_size.extend(size[dim:])
    return tensor.unsqueeze(dim).expand(new_size)


def sequence_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = lengths.max()

    mask = torch.arange(0, maxlen,
                        dtype=lengths.dtype, device=lengths.device)
    return mask < lengths.unsqueeze(-1)


def truncate_sequences(sequences, size):
    for index, sequence in enumerate(sequences):
        sequences[index] = sequence[:size]


def iter_batch_spans(size, batch_size):
    for i in range((size + batch_size - 1) // batch_size):
        yield i * batch_size, min(size, (i + 1) * batch_size)


def pad_2d_values(in_values, dim1=None, dim2=None, dtype=np.int64):
    if dim1 is None or dim2 is None:
        dim1 = len(in_values)
        dim2 = max(len(x) for x in in_values)
    out_values = np.zeros((dim1, dim2), dtype=dtype)
    dim1 = min(len(in_values), dim1)
    for i in range(dim1):
        values = in_values[i]
        current_dim2 = min(dim2, len(values))
        out_values[i, :current_dim2] = values[:current_dim2]
    return torch.from_numpy(out_values)


def pad_3d_values(in_values, dim1=None, dim2=None, dim3=None, dtype=np.int64):
    if dim1 is None or dim2 is None or dim3 is None:
        dim1 = len(in_values)
        dim2 = max(len(x) for x in in_values)
        dim3 = 0
        for value in in_values:
            dim3 = max(dim3, max(len(x) for x in value))
    out_values = np.zeros((dim1, dim2, dim3), dtype=dtype)
    dim1 = min(dim1, len(in_values))
    for i in range(dim1):
        values_i = in_values[i]
        current_dim2 = min(dim2, len(values_i))
        for j in range(current_dim2):
            values_ij = values_i[j]
            current_dim3 = min(dim3, len(values_ij))
            out_values[i, j, :current_dim3] = values_ij[:current_dim3]
    return torch.from_numpy(out_values)
