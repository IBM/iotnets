import networks.networks as networks
from utils import latency_run, demo_info


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
    print("*"*100)
    print("Create networks and runs a few of them by using dummy input data")
    print("Networks are defined in PACKAGE \"networks\"")
    print("Factory routines are defined in \"networks/networks.py\"")
    print("Examples and checks are in \"networks/helper/check_nets.py\"")
    print("*"*100)
    demo_info()
    print("*"*100)

    demo()

    print("ALL DONE")
