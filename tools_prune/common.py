import torch
from tools_prune.models import resnet_cifar, efficientnet
from . import algo
import torchvision.models as models
import scipy.io as io
import numpy as np
import time
import warnings

import torchsparse

from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)

device = torch.device("cuda:0")

rgb_avg = [0.5, 0.5, 0.5]  # [0.485, 0.456, 0.406]
rgb_std = [0.5, 0.5, 0.5]  # [0.229, 0.224, 0.225]
PARAMETRIZED_MODULE_TYPES = (
    torch.nn.Linear,
    torchsparse.nn.Conv3d,
)
NORM_MODULE_TYPES = (torch.nn.BatchNorm2d, torch.nn.LayerNorm)


def loadnetwork(archname, gpuid, act_bitwidth=-1):
    global device
    device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")
    if archname in resnet_cifar.__all__:
        net = resnet_cifar.__dict__[archname](pretrained=True)

    return net.to(device)


def replaceconv(net, layers, includenorm=True):
    pushconv([layers], net, includenorm, direction=1)
    return net


def findconv(net, includenorm=True):
    layers = pushconv([[]], net, includenorm)
    return layers


def getdevice():
    global device
    return device


def pushattr(layers, container, attr, includenorm, direction, prefix=""):
    if isinstance(getattr(container, attr, None), PARAMETRIZED_MODULE_TYPES) or (
        isinstance(getattr(container, attr, None), NORM_MODULE_TYPES) and includenorm
    ):
        if direction == 0:
            layers[0].append(getattr(container, attr))
        else:
            setattr(container, attr, layers[0][0])
            layers[0] = layers[0][1 : len(layers[0])]


def pushlist(layers, container, attr, includenorm, direction, prefix=""):
    if isinstance(container[attr], PARAMETRIZED_MODULE_TYPES) or (
        isinstance(container[attr], NORM_MODULE_TYPES) and includenorm
    ):
        # container[attr] = TimeWrapper(container[attr], prefix)
        if direction == 0:
            layers[0].append(container[attr])
        else:
            container[attr] = layers[0][0]
            layers[0] = layers[0][1 : len(layers[0])]
    else:
        pushconv(layers, container[attr], includenorm, direction, prefix=prefix)


def pushconv(layers, container, includenorm=True, direction=0, prefix="model"):
    if isinstance(container, models.densenet.DenseNet):
        pushconv(layers, container.features, includenorm, direction)
        pushattr(layers, container, "classifier", includenorm, direction)
    elif isinstance(container, models.densenet._DenseBlock):
        for l in range(0, 25):
            if hasattr(container, "denselayer%d" % l):
                pushconv(layers, getattr(container, "denselayer%d" % l), includenorm, direction)
    elif isinstance(container, models.densenet._DenseLayer):
        pushattr(layers, container, "conv1", includenorm, direction)
        pushattr(layers, container, "norm2", includenorm, direction)
        pushattr(layers, container, "conv2", includenorm, direction)
    elif isinstance(container, models.densenet._Transition):
        pushattr(layers, container, "norm", includenorm, direction)
        pushattr(layers, container, "conv", includenorm, direction)
    elif isinstance(container, (torch.nn.Sequential, torch.nn.ModuleList)):
        for attr in range(0, len(container)):
            pushlist(layers, container, attr, includenorm, direction, prefix=prefix + f".{attr}")

    elif isinstance(container, resnet_cifar.ResNetCifar):
        pushattr(layers, container, "conv1", includenorm, direction)
        pushattr(layers, container, "bn1", includenorm, direction)
        pushconv(layers, container.layer1, includenorm, direction)
        pushconv(layers, container.layer2, includenorm, direction)
        pushconv(layers, container.layer3, includenorm, direction)
        pushattr(layers, container, "fc", includenorm, direction)
    elif isinstance(container, resnet_cifar.BasicBlock):
        pushattr(layers, container, "conv1", includenorm, direction)
        pushattr(layers, container, "bn1", includenorm, direction)
        pushattr(layers, container, "conv2", includenorm, direction)
        pushattr(layers, container, "bn2", includenorm, direction)
        if container.downsample is not None:
            pushattr(layers, container.downsample, "0", includenorm, direction)
            pushattr(layers, container.downsample, "1", includenorm, direction)
    else:
        # print("in else")
        return [m for m in container.modules() if isinstance(m, PARAMETRIZED_MODULE_TYPES)]

    return layers[0]


def replacelayer(module, layers, classes):
    module_output = module
    # base case
    if isinstance(module, classes):
        module_output, layers[0] = layers[0][0], layers[0][1:]
    # recursive
    for name, child in module.named_children():
        module_output.add_module(name, replacelayer(child, layers, classes))
    del module
    return module_output


def loadvarstats(archname, testsize):
    mat = io.loadmat(("%s_stats_%d.mat" % (archname, testsize)))
    return np.array(mat["cov"])


def findrdpoints(y_sse, delta, coded, lam_or_bit, is_bit=False):
    # find the optimal quant step-size
    y_sse[np.isnan(y_sse)] = float("inf")
    ind1 = np.nanargmin(y_sse, 1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1, 1).repeat(ind1.shape[1], 1)
    ind2 = np.arange(ind1.shape[1]).reshape(1, -1).repeat(ind1.shape[0], 0)
    inds = np.ravel_multi_index((ind0, ind1, ind2), y_sse.shape)  # bit_depth x blocks
    y_sse = y_sse.reshape(-1)[inds]
    delta = delta.reshape(-1)[inds]
    coded = coded.reshape(-1)[inds]
    # mean = mean.reshape(-1)[inds]
    # find the minimum Lagrangian cost
    if is_bit:
        point = coded == lam_or_bit
    else:
        point = y_sse + lam_or_bit * coded == (y_sse + lam_or_bit * coded).min(0)
    return np.select(point, y_sse), np.select(point, delta), np.select(point, coded)  # , np.select(point, mean)


def predict(net, images, batch_size=256, num_workers=16):
    global device
    y_hat = torch.zeros(0, device=device)
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_hat = torch.cat((y_hat, net(x)))
            if len(y_hat) >= calib_size:
                break
    return y_hat


def predict2(net, loader):
    global device
    # y_hat = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    # with torch.no_grad():
    #     for x, _ in iter(loader):
    #         x = x.to(device)
    #         y_hat = torch.cat((y_hat, net(x)))
    # return y_hat
    # y_hat = []
    # net.eval()
    # with torch.no_grad():
    #     for data in loader:
            
    #         y_hat = torch.cat((y_hat, torch.tensor(net(data)["point_predict"])))
    #         # print(y_hat)
    #         # y_gt = torch.cat((y_gt, net(data)["point_labels"]))
    #         # y_hat.append(net(data)["point_predict"])
    #         # y_gt.append(net(data)["point_labels"])
    # return y_hat
    y_hat = []
    max_len = 0
    # net.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            # print(data)
            
            tmp = net(data)
            y_hat.extend(np.array(tmp["point_predict"]).ravel())
            # break
        ## PADDING---
        # print(y_hat)
        for i in y_hat:
            if len(i) > max_len:
                max_len = len(i)
        
        for i in range(len(y_hat)):
            y_hat[i] = np.append(y_hat[i], np.zeros(shape = max_len - len(y_hat[i])))
        
        y_hat = np.array(y_hat)        
        return y_hat


def predict2_withgt(net, loader):
    global device
    # y_hat = torch.zeros(0, device=device)
    # y_gt = torch.zeros(0, device=device)
    # # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    # with torch.no_grad():
    #     for x, y in iter(loader):
    #         x = x.to(device)
    #         y = y.to(device).float()
    #         y_hat = torch.cat((y_hat, net(x)))
    #         y_gt = torch.cat((y_gt, y))
    # return y_hat, y_gt

    y_hat = []
    y_gt = []
    max_len = 0
    # net.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            # print(data)
            tmp = net(data)
            y_hat.extend(np.array(tmp["point_predict"]).ravel())
            y_gt.extend(np.array(tmp["point_labels"]).ravel())
            # break
        ## PADDING---
        for i in y_gt:
            if len(i) > max_len:
                max_len = len(i)
        
        for i in range(len(y_gt)):
            # print(max_len, len(y_hat[i]))
            y_hat[i] = np.append(y_hat[i], np.zeros(shape = max_len - len(y_hat[i])))
            y_gt[i] = np.append(y_gt[i], np.zeros(shape = max_len - len(y_gt[i])))

        y_hat , y_gt = np.array(y_hat), np.array(y_gt)
        # print(y_hat.shape, y_gt.shape)

    return y_hat, y_gt

def predict2_activation(net, loader, layerhook):
    global device
    acts = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in iter(loader):
            x = x.to(device)
            _ = net(x)
            acts = torch.cat((acts, layerhook.output_tensor))
    return acts


def predict_dali(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            res = net(x)
            y_hat = torch.cat((y_hat, res))
    return y_hat


def predict_dali_withgt(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    y_gt = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            y_hat = torch.cat((y_hat, net(x)))
            y = data[0]["label"]
            y_gt = torch.cat((y_gt, y))
    return y_hat, y_gt
    


def predict_dali_activation(net, loader, layerhook):
    global device
    acts = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            _ = net(x)
            acts = torch.cat((acts, layerhook.output_tensor))
    return acts


import math


@torch.no_grad()
def predict_tensor(net, X, batchsize=128):
    global device
    y_hat = torch.zeros(0, device=device)
    for b in range(math.ceil(len(X) / batchsize)):
        y_hat = torch.cat((y_hat, net(X[b * batchsize : (b + 1) * batchsize])))
    return y_hat


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # pred.reshape(pred.shape[0], -1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def binary_search(min_val, max_val, target_func, target_val, epsilon=0.02, max_iters=40):
    l = min_val
    r = max_val
    cnt = 0
    while l < r:
        mid = (l + r) / 2
        y_mid = target_func(mid)

        if abs(y_mid - target_val) <= epsilon:
            return mid
        elif y_mid < target_val:
            l = mid
        elif y_mid > target_val:
            r = mid

        cnt += 1
        if cnt >= max_iters:
            y_l = target_func(l)
            y_r = target_func(r)
            if abs(y_mid - target_val) > abs(y_l - target_val) and abs(y_r - target_val) > abs(y_l - target_val):
                mid = l
            elif abs(y_mid - target_val) > abs(y_r - target_val) and abs(y_l - target_val) > abs(y_r - target_val):
                mid = r
            break
    return mid


from functools import partial


def find_slope(
    model, target_sparsity, rd_dist, rd_amount, layers=None, prune_mode="unstructured", flop_budget=False, **kwargs
):
    layers = layers or findconv(model, False)
    if flop_budget:
        layer_weights = [layer.weight.clone() for layer in layers]

    def target_func(slope):
        if not flop_budget:
            total_n_weights = 0
            survived_n_weights = 0
        else:
            layer_weights = [layer.weight.clone() for layer in layers]
        pc_amount = algo.pareto_condition(layers, rd_dist, rd_amount, slope)

        for i in range(0, len(layers)):
            prune_weights = algo.pruning(
                layers[i].weight.clone(),
                pc_amount[i][0],
                mode=prune_mode,
                rank=kwargs.get("rank", "l1"),
            )
            if not flop_budget:
                total_n_weights += prune_weights.numel()
                survived_n_weights += (prune_weights != 0).sum().float()
            else:
                layers[i].weight.data = prune_weights

        if flop_budget:
            flops = get_model_flops(model, kwargs["dataset"])
            for layer, ori_weight in zip(layers, layer_weights):
                layer.weight.data = ori_weight
            print(slope, flops)
            return -flops
        ret = 1 - survived_n_weights / total_n_weights
        # print(pc_amount, slope, ret)
        
        return ret

    return binary_search(-1000, 1000, target_func, -target_sparsity if flop_budget else target_sparsity)


def prune_layerbylayer(model, layer_id, amount):
    layers = findconv(model, False)
    layers[layer_id].amount = amount


def hooklayers(layers):
    return [Hook(layer) for layer in layers]


class Hook:
    def __init__(self, module, backward=False):
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # print(type(input[0]), type(output))
        # print(isinstance(input[0], torch.Tensor))
        if not isinstance(input[0], torch.Tensor):
            self.input_tensor = input[0]
            # self.input_tensor = torch.tensor(input)
            # self.output = torch.tensor(output)
            # print(self.input_tensor)
            # print(dir(self.input_tensor))
            # print(self.input_tensor.s)
            # print(input[0])
            # print(output[0])
            self.input =  self.input_tensor.s[1:] # torch.tensor(input[0].shape[1:])
            self.output = output.s[1:] # torch.tensor(output[0].shape[1:])
        # self.input_tensor = input[0]
        # self.output_tensor = output
        else:
            # print('here!!!!!')
            self.input_tensor = input[0]
            # self.input_tensor = torch.tensor(input)
            # self.output = torch.tensor(output)
            # print(self.input_tensor)
            # print(dir(self.input_tensor))
            # print(self.input_tensor.s)
            # print(input[0])
            # print(output[0])
            self.input =  self.input_tensor.size()[1] # torch.tensor(input[0].shape[1:])
            self.output = output.size()[1] # torch.tensor(output[0].shape[1:])

    
    def close(self):
        self.hook.remove()


from tools_prune.pruners import get_weights, get_modules
import torchsparse.nn as spnn

@torch.no_grad()
def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        if isinstance(m, spnn.Conv3d):
            unmaskeds.append(m.kernel.count_nonzero())
        else:
            unmaskeds.append(m.weight.count_nonzero())
    return torch.FloatTensor(unmaskeds)


@torch.no_grad()
def _count_total_weights(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    numels = []
    for w in wlist:
        numels.append(w.numel())
    return torch.FloatTensor(numels)


@torch.no_grad()
def get_model_flops(net, dataset="cifar", custom_data = None):
    # print('here')
    # print(net)
    net.eval()
    if not custom_data:
        if dataset == "cifar":
            dummy_input = torch.zeros((1, 3, 32, 32), device=next(net.parameters()).device)
        else:
            dummy_input = torch.zeros((1, 3, 224, 224), device=next(net.parameters()).device)
    else:
        dummy_input = next(iter(custom_data))


    layers = findconv(net, False)
    unmaskeds = _count_unmasked_weights(net)
    totals = _count_total_weights(net)

    hookedlayers = hooklayers(layers)
    
    # print(len(layers), len(unmaskeds), len(totals))
    # print(unmaskeds, totals)
    _ = net(dummy_input)
    
    fil = [hasattr(h, "output") for h in hookedlayers]
    # print(fil)
    
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
        hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]
        unmaskeds = [unmaskeds[i] for i in range(len(unmaskeds)) if fil[i]]
        totals = [totals[i] for i in range(len(totals)) if fil[i]]
    
    output_dimens = [hookedlayers[i].output for i in range(0, len(hookedlayers))]
    
    # print(output_dimens)
    for l in hookedlayers:
        l.close()
    # exit()
    denom_flops = 0.0
    nom_flops = 0.0

    for o_dim, surv, tot, m in zip(output_dimens, unmaskeds, totals, layers):
        # print(type(m))
        # print(f"{o_dim} \t {surv}\t {tot}\t {m}")
        if isinstance(m, torchsparse.nn.Conv3d):
            # print(m)
            # denom_flops += o_dim[-2:].prod() * tot + (0 if m.bias is None else o_dim.prod())
            # nom_flops += o_dim[-2:].prod() * surv + (0 if m.bias is None else o_dim.prod())
            denom_flops += (o_dim[-2] * o_dim[-1]) * int(tot) + (0 if m.bias is None else o_dim.prod())
            nom_flops += (o_dim[-2] * o_dim[-1]) * int(surv) + (0 if m.bias is None else o_dim.prod())
        elif isinstance(m, torch.nn.Linear):
            # print(f"{o_dim} \t {surv}\t {tot}\t {m}")
            # print(m.bias is None)
            # denom_flops += tot + (0 if m.bias is None else o_dim.prod())
            # nom_flops += surv + (0 if m.bias is None else o_dim.prod())

            denom_flops += tot 
            nom_flops += surv 
            
    # print(nom_flops)
    return nom_flops / denom_flops
