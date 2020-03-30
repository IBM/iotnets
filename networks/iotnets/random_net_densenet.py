import math
import numpy as np
import torch

from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.densenet import Bottleneck
from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.densenet import DenseNet


def get_config(space):
    config = {}
    if space == "s1":  # regular
        config["nblocks"] = [1, 32]
        config["growth_rate"] = [1, 32]
        config["reduction"] = [20, 80]
    elif space == "s2":  # upper to 16
        config["nblocks"] = [1, 16]
        config["growth_rate"] = [1, 16]
        config["reduction"] = [20, 80]
    elif space == "s3":  # upper to 8
        config["nblocks"] = [1, 8]
        config["growth_rate"] = [1, 8]
        config["reduction"] = [50, 80]
    else:
        raise ValueError("Space {} not defined!".format(space))

    return config


def sample(config, id):
    rstate = np.random.RandomState(id)
    nblocks = rstate.randint(*config["nblocks"], 4)
    growth_rate = int(rstate.randint(*config["growth_rate"], 1))
    reduction = float(rstate.randint(*config["reduction"], 1)/100.0)

    # nblocks 1-32, growth_rate 1-32, reduction 20 - 80 (integer, in %)
    return dict(nblocks=nblocks, growth_rate=growth_rate, reduction=reduction)


def get_instance(net_args,num_classes):
    return DenseNet(Bottleneck, num_classes=num_classes, **net_args)

# This stub is required to find this file as module available!
class dummy():
    pass
