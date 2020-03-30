import networks.networks as networks
from main_run_dummy import latency_run, demo_info
from utils import load_json, write_json

import numpy as np
import torch
import time
import os

from common import load_net


def run_model(config, root_models, device="cpu"):
    # Defines the model topology
    law_config = networks.get_law_config(config["spacename"], config["law"])
    net_args = networks.sample_from_law(config["spacename"], law_config, config["id"])
    net = networks.build(config["spacename"], 10, net_args)

    net = net.to(device)

    # Defines where the model is stored
    ckpt_model = os.path.join(root_models, "{}_ckpt.t7".format(config["runid"]))

    # Loading model
    mode = "None" if device == "cuda" else "rm"
    net, acc = load_net(net, ckpt_model, device, mode=mode, verbose=False)
    print("Loaded net acc: {}".format(acc))

    # Latency measurement
    ts = latency_run(net, device="cpu", bs=1, rep=10, verbose=True)
    return np.mean(1000 * ts[5:])


if __name__ == '__main__':
    print("*"*100)
    print("Reads and runs trained models")
    print("Requirements:")
    print(" - 1: Download and extract model files (large!)")
    print(" - 2: supply the root folder to this script to find the files")
    print("*"*100)
    demo_info()
    print("*"*100)

    # Marked and relevant models.
    selPICNets = [687, 611, 450, 405, 306, 349, 64]
    selMACCNets = [1552, 1420, 1449, 851, 771, 64]

    # Open full configuration file
    configs = load_json("configs/NetConfigs.json")
    # Define the path to the models
    root_models = "/Users/eid/Desktop/GIT/tpml/ml_plots/PAPER/results/models"

    # MacBook nets
    for i in range(len(selMACCNets)):
        config = configs[selMACCNets[i]]  # select one.
        run_model(config, root_models)

    # Pi-nets
    for i in range(len(selPICNets)):
        config = configs[selPICNets[i]]  # select one.
        run_model(config, root_models)

    print("ALL DONE")
