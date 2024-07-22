Added missing “OUTPUT_DIR” to config file for inference.


Important changes to “vis_SemanticKITTI.py” file:
Added the path to plot map of predicted labels(numpy).
Inversion of the predicted labels was a must, as the predictions were inverse.
#### CHANGE
    raw_label = np.vectorize(learning_map_inv.__getitem__)(raw_label)
####								
This line was added right before mapping the objects from the dictionary.



### PRUNER
Copy “spvcnn.py” from UniSeg to tools_prune/models/fusion

In "iterate.py"

Add imports:
```
from pcseg.data import build_dataloader
import yaml
from easydict import EasyDict

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
```
In the model_and_opt_loader add the following:
```

import yaml
from easydict import EasyDict
from .models.fusion.spvcnn.spvcnn import SPVCNN


if model_string == "fusion":
        cfgs = cfg_from_yaml_file("/mnt/e/PCSeg/tools_prune/models/fusion/spvcnn/spvcnn_mk18_cr10.yaml")
        model = SPVCNN(cfgs["MODEL"], num_class = 20)
        amount = 0.20
        batch_size = cfgs["OPTIM"]["BATCH_SIZE_PER_GPU"]
        opt_pre = {}
        opt_post = {"steps": 40000}

        ...

        checkpoint = torch.load(f"{PATH TO YOUR BUILT MODEL}")

```

In prune_weights_reparam method of the above code segment check for instances:
```
if isinstance(m, spnn.Conv3d):
            prune.identity(m, name="kernel")
```
###
Open the config file in “iterate.py”
```
cfgs = cfg_from_yaml_file("/mnt/e/PCSeg/tools_prune/models/fusion/spvcnn/spvcnn_mk18_cr10.yaml")
_, train_loader, _ = build_dataloader(
            data_cfgs=cfgs.DATA,
            modality=cfgs.MODALITY,
            batch_size=1, #cfgs.OPTIM.BATCH_SIZE_PER_GPU,
            dist=cfgs["MODEL"]["IF_DIST"],
            root_path=cfgs.DATA.DATA_PATH,
            workers=5,
            #logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=epoch_cnt)
```

In “iterate.py” change the method weight_pruner_loader()added a new input parameter: “custom_data=None” and pass “train_loader” to it along with default arguments.

###
In common.get_model_flops():
 add the same parameter “custom_data=None” and pass train_loader to it.

###
If statement to check if custom_data is used.

Changed dummy_input to:
```
dummy_input = next(iter(custom_data))
```
###
In PARAMETRIZED_MODULE_TYPES:
Add torchsparse.nn.Conv3d,


In predict2_withgt:
```
global device
    y_hat = []
    y_gt = []
    with torch.no_grad():
        for data in loader:
            y_hat.append(net(data)["point_predict"])
            y_gt.append(net(data)["point_labels"])
    return y_hat, y_gt
```

Changes in the “Hook” class:
In “hook_fn”:
```
if not isinstance(input[0], torch.Tensor):
   self.input_tensor = input[0]
   self.input =  self.input_tensor.s[1:]
   self.output = output.s[1:] 
else:
    self.input_tensor = input[0]
    self.input =  self.input_tensor.size()[1] 
    self.output = output.size()[1] # torch.tensor(output[0].shape[1:])
```
At the bottom:
```
for o_dim, surv, tot, m in zip(output_dimens, unmaskeds, totals, layers):
        if isinstance(m, torchsparse.nn.Conv3d):
            denom_flops += (o_dim[-2] * o_dim[-1]) * int(tot) + (0 if m.bias is None else o_dim.prod())
            nom_flops += (o_dim[-2] * o_dim[-1]) * int(surv) + (0 if m.bias is None else o_dim.prod())
        elif isinstance(m, torch.nn.Linear):
            denom_flops += tot
            nom_flops += surv
    return nom_flops / denom_flops
```
In “algo.py” in the function get_ele_per_output_channel:
```
    # print(tensor_weights.size())
    if len(tensor_weights.shape) == 3:
        tensor_weights = tensor_weights[..., None, None]
        return tensor_weights[0, :, :, :, :].numel()
```
###

In “pruners.py” joblib was used to save dictionary instead of pickle due to memory limit.

###
Changes made to “utils.py”, “spvcnn.py”, “RDPruner()”, “predict2”, “predict2_withgt”.

###
In “eqlv2.py” under losses folder, removed “_func()” method and used its operation directly, as the model could not be serialized earlier because of this.

