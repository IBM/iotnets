import torch
import time
import numpy as np
import json


def load_json(file, verbose=True):
    if verbose:
        print("opening file {} ... ".format(file), end="")
    with open(file, 'r') as fd:
        config = json.load(fd)
    if verbose:
        print("[OK]")
    return config


def write_json(file_name, config, verbose=True):
    if verbose:
        print("Write {} ...".format(file_name), end="")
    with open(file_name, 'w') as json_file:
        json.dump(config, json_file, sort_keys=True, indent=4)
    if verbose:
        print("OK")


def latency_run(net, device = "cpu", bs=1, rep=10, verbose=True):
    net = net.to(device)
    net.eval()

    ts = []
    with torch.no_grad():
        for r in range(rep):
            x = torch.tensor(np.random.randn(bs, 3, 32, 32).astype(np.float32))
            time_start = time.time()
            y = net(x)
            time_duration = time.time() - time_start
            ts.append(time_duration)

    ts = np.array(ts)

    if verbose:
        print("Times: {}".format(ts))
        print("Avg: {:.1f} ms".format(np.mean(1000*ts)))
        print("Std: {:.1f} ms".format(np.std(1000*ts)))
        mid = int(rep/2)
        print("Last Half Avg: {:.1f} ms".format(np.mean(1000*ts[mid:])))
        print("Last Half Std: {:.1f} ms".format(np.std(1000*ts[mid:])))
        print("")

    return ts


def demo_info():
    print("TORCH.__version__: {}".format(torch.__version__))