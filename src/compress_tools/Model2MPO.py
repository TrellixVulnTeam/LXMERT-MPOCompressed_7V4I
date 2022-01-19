import os
from torch import nn
from compress_tools.Linear2MPO import Linear2MPO
from torch import nn


def Model2Mpo(module, prefix='', exclude_module=""):
    for name, child in module._modules.items():
        if name in exclude_module:
            continue
        if child is not None:
            newmodule = getattr(module, name)
            if isinstance(newmodule, nn.Linear):
                mpo_module = Linear2MPO(newmodule, tensor_learn=True)
                mpo_module.from_pretrained(newmodule)
                setattr(module, name, mpo_module)
            Model2Mpo(child, prefix + name + '.')
    
