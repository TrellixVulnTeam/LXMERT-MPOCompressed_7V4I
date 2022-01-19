import numpy as np
from torch import nn as nn
import torch
import logging
import os

from torch._C import default_generator
from compress_tools.Matrix2MPO_beta import MPO
from torch.nn import functional as F
from torch import nn


logger = logging.getLogger(__name__)
DefaultMpoSet = {
    4: [1, 2, 2, 1, 1],
    128: [2, 2, 4, 4, 2],
    256: [2, 4, 4, 4, 2],
    512: [2, 4, 8, 4, 2],
    768: [3, 4, 4, 4, 4],
    1024: [4, 4, 8, 4, 2],
    2048: [4, 4, 8, 4, 4],
    3072: [4, 4, 8, 6, 4],
    30000: [5, 10, 10, 10, 6],
}


class Linear2MPO(nn.Module):
    '''
    compress using MPO method
    ref: Compressing deep neural networks by matrix product operators
    '''

    def __init__(self, linear, mpo_input_shape=None, mpo_output_shape=None, trunc_num=1e7,
                 tensor_learn=True,
                 *args,
                 **kwargs
                 ):
        super(Linear2MPO, self).__init__()
        self.mpo_input_shape = mpo_input_shape
        self.mpo_output_shape = mpo_output_shape
        self.tensor_learn = tensor_learn
        self.tensor_set = None
        self.trunc_num = trunc_num
        self.use_default = mpo_input_shape == None
        self.mpo = None
        self.linear_input_shape = linear.weight.shape[0]
        self.linear_output_shape = linear.weight.shape[1]
        self.bias = linear.bias

    def get_default_mpo(self):
        linear_input_shape = self.linear_input_shape
        linear_output_shape = self.linear_output_shape
        if linear_input_shape not in DefaultMpoSet.keys():
            print("Linear input shape 0 has no default Mpo setting")
            assert(0)
        if linear_output_shape not in DefaultMpoSet.keys():
            print("Linear input shape 1 has no default Mpo setting")
            assert(0)
        self.mpo_input_shape = DefaultMpoSet[linear_input_shape]
        self.mpo_output_shape = DefaultMpoSet[linear_output_shape]

    def get_mpo(self):
        if self.use_default:
            self.get_default_mpo()
        if not self.mpo:
            self.mpo = MPO(self.mpo_input_shape,
                           self.mpo_output_shape, self.trunc_num)

    # def forward(self, x):
    #     # tensor_set = [tensor.cuda() for tensor in self.tensor_set]
    #     input = torch.transpose(x, 0, 1)
    #     input = input.reshape(self.mpo_output_shape + [x.shape[0]])
    #     output = torch.tensordot(
    #         self.tensor_set[0].squeeze(0), input, ([-2], [0]))
    #     for i in range(1, len(self.tensor_set)):
    #         output = torch.tensordot(
    #             self.tensor_set[i], output, ([-2, 0], [i+1, 1]))
    #     output = output.squeeze(1).reshape(
    #         [self.origin_weight.shape[0], x.shape[0]]).transpose(0, 1)
    #     return output

    def forward(self, x):
        mpo = MPO(self.mpo_input_shape, self.mpo_output_shape, self.trunc_num)
        res = x.reshape(-1, x.shape[-1])
        res = F.linear(res, mpo.mpo2matrix(self.tensor_set), self.bias)
        ori_shape = x.shape
        return res.view((tuple(ori_shape[:-1])+(-1,)))

    def from_pretrained(self, linear):
        self.get_mpo()
        tensor_set, _, _ = self.mpo.matrix2mpo(
            linear.weight.data.cpu().numpy())
        self.tensor_set = torch.nn.ParameterList([nn.Parameter(
            torch.from_numpy(i).cuda(), requires_grad=True) for i in tensor_set])
        if self.tensor_learn:
            self.tensor_set[2].requires_grad = False
        # if self.origin_bias != None:
        #     self.bias = self.origin_bias
        else:
            logger.info("Check no bias")
            self.bias = None
