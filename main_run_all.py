import numpy as np

import networks.networks as networks
from main_run_dummy import latency_run, demo_info
from utils import load_json, write_json


def run_all():
    # Open configuration file
    configs = load_json("configs/NetConfigs.json")

    results = []
    # Run all configurations
    for c in configs:
        law_config = networks.get_law_config(c["spacename"], c["law"])
        net_args = networks.sample_from_law(c["spacename"], law_config, c["id"])
        net = networks.build(c["spacename"], 10, net_args)
        ts = latency_run(net, device = "cpu", bs=1, rep=10, verbose=True)
        # Clean the data
        c["t_avg"] = np.mean(1000*ts[5:])
        c["t_std"] = np.std(1000*ts[5:])
        results.append(c)

    # Save results (with configurations)
    write_json("results.json", results)


if __name__ == '__main__':
    print("*"*100)
    print("Reads all configurations from \"configs/NetConfigs.py\"")
    print("Runs and measures the latency")
    print("Saves the results into result.json")
    print("*"*100)
    demo_info()
    print("*"*100)

    run_all()

    print("ALL DONE")
