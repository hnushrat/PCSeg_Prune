import torch
import torch.nn as nn
from copy import deepcopy

import torchsparse.nn as spnn

def _is_prunable_module(m):
    # return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d))
    return (isinstance(m,nn.Linear) or isinstance(m,spnn.Conv3d))

def _get_sparsity(tsr):
    total = tsr.numel()
    nnz = tsr.nonzero().size(0)
    return nnz/total
    
def _get_nnz(tsr):
    return tsr.nonzero().size(0)


def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m):
            if isinstance(m, spnn.Conv3d):
                weights.append(m.kernel)
            else:
                weights.append(m.weight)
    return weights

def get_convweights(model):
    weights = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            weights.append(m.weight)
    return weights

def get_modules(model):
    modules = []
    for m in model.modules():
        # print(isinstance(m, nn.Conv3d))
        if _is_prunable_module(m):
            # print(f"*** {m}")
            modules.append(m)
    return modules

def get_convmodules(model):
    modules = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            modules.append(m)
    return modules

def get_copied_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(deepcopy(m).cpu())
    return modules

def get_model_sparsity(model):
    prunables = 0
    nnzs = 0
    for m in model.modules():
        if _is_prunable_module(m):
            if isinstance(m, spnn.Conv3d):
                prunables += m.kernel.data.numel()
                nnzs += m.kernel.data.nonzero().size(0)
            else:
                prunables += m.weight.data.numel()
                nnzs += m.weight.data.nonzero().size(0)
    return 1 - nnzs/prunables

def get_sparsities(model):
    return [_get_sparsity(m.weight.data) for m in model.modules() if _is_prunable_module(m)]

def get_nnzs(model):
    return [_get_nnz(m.weight.data) for m in model.modules() if _is_prunable_module(m)]
