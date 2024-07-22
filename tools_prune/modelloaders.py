import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.models as tmodels
from functools import partial
from tools_prune.models import *
import tools_prune.models.resnet_cifar as resnet_cifar
from tools_prune.pruners import prune_weights_reparam


####################################################################
import yaml
from easydict import EasyDict
# from .models.fusion.spvcnn.spvcnn import SPVCNN #, MinkUNet
from pcseg.model.segmentor.fusion.spvcnn.spvcnn import SPVCNN
from torch.cuda import amp

from pcseg.optim import build_optimizer, build_scheduler

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config = EasyDict()):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)
        
        # print(new_config)####

        merge_new_config(config=config, new_config=new_config)

    return config


####################################################################



def model_and_opt_loader(model_string,DEVICE, reparam=True, weight_rewind=False, loader = None):

    if DEVICE == None:
        raise ValueError('No cuda device!')
    
    if model_string == "fusion":
        cfgs = cfg_from_yaml_file("/mnt/e/PCSeg/tools/cfgs/fusion/semantic_kitti/spvcnn_mk18_cr10.yaml")
        model = SPVCNN(cfgs["MODEL"], num_class = 20)
        # print(model.state_dict())
        # model.load_state_dict()#
        
        optimizer = build_optimizer(
            model=model,
            optim_cfg=cfgs.OPTIM,)
        
        scaler = amp.GradScaler(enabled=False)

        scheduler = build_scheduler(
            optimizer,
            total_iters_each_epoch=len(loader),
            total_epochs=36,
            optim_cfg=cfgs["OPTIM"],)
        
        checkpoint = torch.load("/mnt/e/PCSeg/logs/fusion_unpruned/semantic_kitti/spvcnn_mk18_cr10/default/ckp/checkpoint_epoch_36.pth")

        model.load_params(checkpoint['model_state'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        # print(model.state_dict())
        print('loaded!!!')
        
        amount = 0.70 # 0.20
        batch_size = cfgs["OPTIM"]["BATCH_SIZE_PER_GPU"]
        opt_pre = {}
        opt_post = {"steps": 40000}

    elif model_string == 'vgg16':
        model = VGG16().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 20000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'resnet18':
        model = ResNet18().to(DEVICE)
        amount = 0.2
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string in resnet_cifar.__dict__:
        model = resnet_cifar.__dict__[model_string]().to(DEVICE)
        amount = 0.2
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 50000,
            "scheduler": partial(sched.CosineAnnealingLR, T_max=50000)
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 80000,
            "scheduler": None
        }
    elif model_string == 'densenet':
        model = DenseNet121().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 80000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 60000,
            "scheduler": None
        }
    elif model_string == 'effnet':
        model = EfficientNetB0().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'vgg16_bn':
        model = vgg16_bn(True).to(DEVICE)
        amount = 0.6
        batch_size = 32
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 0,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD if not weight_rewind else optim.Adam,lr=0.01, weight_decay=0.0001),
            # "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 80000,
            "scheduler": partial(sched.CosineAnnealingLR, T_max=80000)
        }
    elif model_string == 'resnet50':
        model = resnet50(True).to(DEVICE)
        amount = 0.2
        batch_size = 64
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 0,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD if not weight_rewind else optim.Adam,lr=0.01, weight_decay=0.0001),
            # "optimizer": partial(optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": partial(sched.CosineAnnealingLR, T_max=40000)
        }
    else:
        raise ValueError(f'Unknown model: {model_string}')
    if reparam:
        prune_weights_reparam(model)
    return model,amount,batch_size,opt_pre,opt_post #, optimizer,scaler,scheduler
