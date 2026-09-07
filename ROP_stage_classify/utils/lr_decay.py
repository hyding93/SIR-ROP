import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):

    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())



def param_groups_lrd_rip(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    param_group_names = {}
    param_groups = {}

    num_layers = 15

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_rip(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_rip(name, num_layers):
    if name.startswith('head') or name.startswith('classifier'):
        return num_layers
    else:
        try:
            num_layers = int(name.split('.')[1]) + int(name.split('.')[2]) + 2
        except:
            num_layers = int(name.split('.')[2]) + int(name.split('.')[3]) + 2
        
        return num_layers



def get_layer_id_for_vit(name, num_layers):
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers








def param_groups_lrd_rip_v2(model, weight_decay=0.05, layer_decay=0.75):
    param_group_names = {}
    param_groups = {}

    num_layers = 5
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 2)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n.endswith("bias") or "layer_scale" in n:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_rip(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": []
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": []
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())



def get_layer_id_for_rip_v2(name, num_layers=5):
    if name.startswith("head") or name.startswith("classifier"):
        return num_layers + 1
    
    if name.startswith("sn_unet.0"):
        return 0
    
    if name.startswith("sn_unet.1"):
        return 1
    
    if name.startswith("sn_unet.2"):
        return 2
    
    if name.startswith("sn_unet.3"):
        return 3
    
    if name.startswith("sn_unet.4"):
        return 4

    if name.startswith("sn_unet.5"):
        return 5
    
    return num_layers + 1 