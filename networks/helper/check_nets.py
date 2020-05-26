import os
import time
import torch
import numpy as np

p = os.path.abspath('../..')
print(p)

import networks.networks as networks


def simple_resnet():
    netname = "ResNet18"
    net = networks.build(netname, 10)
    print(net)


def all_thirdparty():
    for netname in networks.KUANGLIU_NETS:
        print(netname)
        # Check if you can construct the network.
        net = networks.build(netname, 10)


def all_probenets():
    for netname in networks.PROBE_NETS:
        print(netname)
        # Check if you can construct the network.
        net = networks.build(netname, 10)


def all_main_iotnets_manual():
    import networks.iotnets.random_net_densenet as random_net_densenet

    config = random_net_densenet.get_config("s1")
    net_args = random_net_densenet.sample(config, 2)
    net = random_net_densenet.get_instance(net_args, 12)
    print(net)

    import networks.iotnets.random_net_googlenet as random_net_googlenet

    config = random_net_googlenet.get_config("s1")
    net_args = random_net_googlenet.sample(config, 0)
    net = random_net_googlenet.get_instance(net_args, 12)
    print(net)


    import networks.iotnets.random_net_mobilenetv2 as random_net_mobilenetv2

    config = random_net_mobilenetv2.get_config("s1")
    net_args = random_net_mobilenetv2.sample(config, 0)
    net = random_net_mobilenetv2.get_instance(net_args, 12)
    print(net)

    import networks.iotnets.random_net_pnasnet as random_net_pnasnet

    config = random_net_pnasnet.get_config("s1")
    net_args = random_net_pnasnet.sample(config, 0)
    net = random_net_pnasnet.get_instance(net_args, 12)
    print(net)

    import networks.iotnets.random_net_resnet as random_net_resnet

    config = random_net_resnet.get_config("s1")
    net_args = random_net_resnet.sample(config, 0)
    net = random_net_resnet.get_instance(net_args, 12)
    print(net)

    import networks.iotnets.random_net_resnext as random_net_resnext

    config = random_net_resnext.get_config("s1")
    net_args = random_net_resnext.sample(config, 0)
    net = random_net_resnext.get_instance(net_args, 12)
    print(net)


# nicer way to achieve the same.
def all_iotnets():
    for spacename in networks.ARCH_SPACES:
        print(spacename)
        # Check if you can construct the network.
        law_config = networks.get_law_config(spacename, "s1")
        net_args = networks.sample_from_law(spacename, law_config, 0)
        net = networks.build(spacename, 12, net_args)
    print("OK")


def all_synthnets():
    net_args = dict(f=1)
    net = networks.build("SynthWide", 12, net_args)

    net_args = dict(f=1, l=2)
    net = networks.build("SynthWideAndLong", 12, net_args)

    net_args = dict(f=1)
    net = networks.build("SynthWide256", 12, net_args)



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


def demo():
    for arch in ['ResNet18', 'ResNet101', 'ResNet152']:
        print("REFERENCE model: {}".format(arch))
        net = networks.build(arch, 10, net_args=None)
        ts = latency_run(net)


    spacename = "random_net_mobilenetv2"
    for law in ["s1", "s3", "s6"]:
        print("OUR Constraint ARCH searchspace, based on {} with sampling law {}".format(spacename, law))
        law_config = networks.get_law_config(spacename, law)
        net_args = networks.sample_from_law(spacename, law_config, 0)
        net = networks.build(spacename, 10, net_args)
        ts = latency_run(net)


if __name__ == '__main__':
    print("Create networks in PyTorch")
    # simple_resnet()
    # all_thirdparty()
    # all_probenets()
    all_main_iotnets_manual()
    all_iotnets()
    all_synthnets()

    demo()



    print("ALL DONE")

