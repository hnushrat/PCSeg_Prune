import torch, argparse, random, os
import numpy as np
from tools_prune import *
import tools_prune.common as common


import logging
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from tqdm import tqdm

from pcseg.data import build_dataloader
import yaml
from easydict import EasyDict

""" ARGUMENT PARSING """
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--cuda", type=int, help="cuda number")
parser.add_argument("--model", type=str, help="network")
parser.add_argument("--pruner", type=str, help="pruning method")
parser.add_argument("--dataset", type=str, choices=["cifar", "imagenet"], default="cifar")
parser.add_argument("--iter_start", type=int, default=1, help="start iteration for pruning (set >1 for resume)")
parser.add_argument("--iter_end", type=int, default=20, help="end iteration for pruning")
parser.add_argument("--maxsps", type=int, default=100)
parser.add_argument("--ranking", type=str, default="l1")
parser.add_argument(
    "--prune_mode", "-pm", type=str, default="unstructured", choices=["unstructured", "structured"]
)
parser.add_argument("--calib_size", type=int, default=20)
parser.add_argument("--weight_rewind", action="store_true")
parser.add_argument("--worst_case_curve", "-wcc", action="store_true")
parser.add_argument("--synth_data", action="store_true")
parser.add_argument("--singlelayer", action="store_true")
parser.add_argument(
    "--flop_budget",
    action="store_true",
    help="use flop as the targeting budget in ternary search instead of sparsity. if true, `amounts` and `target_sparsity` variables in the codes will represent flops instead",
)
args = parser.parse_args()

""" SET THE SEED """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

DEVICE = args.cuda

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

cfgs = cfg_from_yaml_file("/mnt/e/PCSeg/tools/cfgs/fusion/semantic_kitti/spvcnn_mk18_cr10.yaml") # path to config file

epoch_cnt = 36

_, train_loader, _ = build_dataloader(
            data_cfgs=cfgs.DATA,
            modality=cfgs.MODALITY,
            batch_size=cfgs.OPTIM["BATCH_SIZE_PER_GPU"],
            dist=cfgs["MODEL"]["IF_DIST"],
            root_path=cfgs.DATA.DATA_PATH,
            workers=5,
            #logger=logger,
            training=False,
            merge_all_iters_to_one_epoch=False,
            total_epochs=epoch_cnt)


# model = torch.load("PRUNED_MODEL_50_percent.pth")

# flops = common.get_model_flops(model, args.dataset, custom_data = train_loader)
# sparse = utils.get_model_sparsity(model)
    


# # print(f"{summary(model)}")
# print(f"sparsity: {sparse * 100} (%)")
# print(f"Remained FLOPs: {flops * 100} (%)")
# for p in model.parameters():
#     print(p)


# exit()


""" IMPORT LOADERS/MODELS/PRUNERS/TRAINERS"""
model, amount_per_it, batch_size, opt_pre, opt_post = model_and_opt_loader(
    args.model, DEVICE, weight_rewind=args.weight_rewind, reparam=True, loader = train_loader
)


# print(model.state_dict())
# print(f"{summary(model)}")
# print(model, amount_per_it, batch_size, opt_pre, opt_post)

# train_loader, test_loader, calib_loader = dataset_loader(args.model, batch_size=batch_size, args=args) # Uncommented initially




pruner = weight_pruner_loader(args.pruner, custom_data = train_loader)

# trainer = trainer_loader(args)


container, _, _, _, _= model_and_opt_loader(args.model, DEVICE, reparam=False, weight_rewind=args.weight_rewind, loader = train_loader)


""" SET SAVE PATHS """
DICT_PATH = f"./pruned/dicts/{args.model}/{args.seed}/{args.prune_mode}/" # modified
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)
BASE_PATH = f"./pruned/results/iterate/{args.model}/{args.seed}/{args.prune_mode}/" # modified
if args.weight_rewind:
    BASE_PATH += "/weight_rewind/"
    DICT_PATH += "/weight_rewind/"
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)


results_to_save = []

'''
""" PRETRAIN (IF NEEDED) """
if args.iter_start == 1:
    filename_string = "unpruned.pth.tar"
else:
    filename_string = args.pruner + str(args.iter_start - 1) + ".pth.tar"
if os.path.exists(os.path.join(DICT_PATH, filename_string)):
    print(f"LOADING PRE-TRAINED MODEL: SEED: {args.seed}, MODEL: {args.model}, ITER: {args.iter_start - 1}")
    state_dict = torch.load(os.path.join(DICT_PATH, filename_string), map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    try:
        results_to_save = [[i.item() for i in l] for l in torch.load(BASE_PATH + f"/{args.pruner}.tsr")]
    except:
        pass
else:
    if args.iter_start == 1:
        print(f"PRE-TRAINING A MODEL: SEED: {args.seed}, MODEL: {args.model}")
        pretrain_results = trainer(model, opt_pre, train_loader, test_loader, print_steps=1000)
        torch.save(pretrain_results, DICT_PATH + "/unpruned_loss.dtx")
        torch.save(model.state_dict(), os.path.join(DICT_PATH, "unpruned.pth.tar"))
    else:
        raise ValueError("No (iteratively pruned/trained) model found!")


# epoch_cnt = args.iter_start * opt_post["steps"] # modified

# opt_post["cutmix_alpha"] = args.cutmix_alpha
# print(args.cutmix_alpha)
'''

# print(pruner)
# print(model)


""" PRUNE AND RETRAIN """
for it in range(args.iter_start, args.iter_end + 1): # 1
    
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    flops = common.get_model_flops(model.to(DEVICE), args.dataset, custom_data = train_loader)
    print(f"Before prune: FLOPs: {flops}")
    if args.pruner == "rd":
        args.remask_per_iter = opt_post["steps"]  # just for naming
        # pruner(model, amount_per_it, args, calib_loader, container, epoch_cnt=epoch_cnt)
        pruner(model.to(DEVICE), amount_per_it, args, train_loader, container.to(DEVICE), epoch_cnt=epoch_cnt, custom_data = train_loader)
    else:
        pruner(model, amount_per_it)

    if args.weight_rewind and os.path.exists(os.path.join(DICT_PATH, args.pruner + str(it - 1) + ".pth.tar")):
        model.load_state_dict(
            {
                k: v
                for k, v in torch.load(os.path.join(DICT_PATH, args.pruner + str(it - 1) + ".pth.tar")).items()
                if "mask" not in k
            },
            strict=False,
        )

    flops = common.get_model_flops(model, args.dataset, custom_data = train_loader)
    sparse = utils.get_model_sparsity(model)
     
    
    
    # print(f"{summary(model)}")
    print(f"sparsity: {sparse * 100} (%)")
    print(f"Remained FLOPs: {flops * 100} (%)")



torch.save(model, f"PRUNED_MODELS/{(args.prune_mode).upper()}/NEW_PRUNED_MODEL_{int(amount_per_it * 100)}_percent.pth") # -> the pruned model to be used for training

# tmp = torch.load(f"PRUNED_MODEL_{amount_per_it * 100}_percent.pth")
# tmp.eval()
# with torch.no_grad():
#     print(tmp(next(iter(train_loader))))
# exit()
'''
    print(f"\nEVALUATION IN PROGRESS...\n")
    ## EVAL ##

    def fast_hist(pred, label, n):
        k = (label >= 0) & (label < n)
        bin_count = np.bincount(
            n * label[k].astype(int) + pred[k], minlength=n ** 2)
    
        return bin_count[:n ** 2].reshape(n, n)
    
    def fast_hist_crop(output, target, unique_label):
        hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
        hist = hist[unique_label + 1, :]
        hist = hist[:, unique_label + 1]
        return hist
    
    def per_class_iu(hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-9)
    
    metric = {}
    metric['hist_list'] = []
    unique_label = np.array(list(range(19)))

    model.eval()
    with torch.no_grad():
        for i, batch_dict in enumerate(tqdm(train_loader)):
            ret_dict = model(batch_dict)
                
            point_predict = ret_dict['point_predict']
            point_labels = ret_dict['point_labels']

            for pred, label in zip(point_predict, point_labels):
                metric['hist_list'].append(fast_hist_crop(pred, label, unique_label))
            
            # print(metric)
    
    
    logger = create_logger("PRUNING_LOG.txt")
    hist_list = metric['hist_list'][:len(train_loader.dataset)]
    iou = per_class_iu(sum(hist_list))
    logger.info('Validation per class iou: ')
    
    logger_tb = SummaryWriter(log_dir=str('tensorboard')) #if cfgs.LOCAL_RANK == 0 else None
    
    class_names = train_loader.dataset.class_names
    prefix = "val"
    cur_epoch = epoch_cnt

    for class_name, class_iou in zip(class_names[1:], iou):
        logger_tb.add_scalar(f"{prefix}/{class_name}", class_iou * 100, cur_epoch+1)
    
    val_miou = np.nanmean(iou) * 100
    logger_tb.add_scalar(f"{prefix}_miou", val_miou, cur_epoch + 1)

    # logger confusion matrix and
    table_xy = PrettyTable()
    table_xy.title = 'Validation iou'
    table_xy.field_names = ["Classes", "IoU"]
    table_xy.align = 'l'
    table_xy.add_row(["All", round(val_miou, 4)])

    for i in range(len(class_names[1:])):
        table_xy.add_row([class_names[i+1], round(iou[i] * 100, 4)])
    logger.info(table_xy)

    dis_matrix = sum(hist_list)
    table = PrettyTable()
    table.title = 'Confusion matrix'
    table.field_names = ["Classes"] + [k for k in class_names[1:]] + ["Points"]
    table.align = 'l'

    for i in range(len(class_names[1:])):
        sum_pixel = sum([k for k in dis_matrix[i]])
        row = [class_names[i + 1]] + [round(k/(sum_pixel +1e-8) * 100, 4) for k in dis_matrix[i]] + [sum_pixel, ]
        table.add_row(row)
    
    logger.info(table)

'''

    
    
    # print(summary(model))
    
    
    # result_log = trainer(model, opt_post, train_loader, test_loader, print_steps=100)
    # result_log.append(get_model_sparsity(model))
    # result_log.append(flops)
    # results_to_save.append(result_log)
    # torch.save(torch.FloatTensor(results_to_save), BASE_PATH + f"/{args.pruner}.tsr")
    # # if args.weight_rewind:
    # torch.save(model.state_dict(), os.path.join(DICT_PATH, args.pruner + str(it) + ".pth.tar"))
