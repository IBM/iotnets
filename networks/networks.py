"""
This file builds the main entry point to define all PyTorch networks.
Networks might be implemented from different sources in different files.
All networks are uniquely identified by a name and a configuration that is specific to the them.

There is one routine to build all networks:
 * build(...):                  builds one network

That routines includes the case to build specific networks,
to slightly change them by adapting the number of classes (last layer change) or
by defining network type specific net_args (a dict, with network specific construction details).
"""

import networks.thirdparty.kuangliu_models as kuangliu_models
import networks.probenets as probenets
import networks.synthnets as synthnets
import networks.iotnets as iotnets


# The following constants were produced with the script helper/valid_arch_generator.py
KUANGLIU_NETS = ['DPN26', 'DPN92', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'PNASNetA', 'PNASNetB', 'PreActResNet101', 'PreActResNet152', 'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'ResNeXt29_2x64d', 'ResNeXt29_32x4d', 'ResNeXt29_4x64d', 'ResNeXt29_8x64d', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34', 'ResNet50', 'SENet18', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'densenet_cifar']
# NUMBER OF VALID NETWORKS: 28
PROBE_NETS = ['ProbeNet_d_feature', 'ProbeNet_d_length', 'ProbeNet_d_mlp', 'ProbeNet_s', 'ProbeNet_s_deep', 'ProbeNet_s_fat', 'ProbeNet_s_nr_deep', 'ProbeNet_s_nr_shallow', 'ProbeNet_s_shallow', 'ProbeNet_s_slim']
# NUMBER OF VALID NETWORKS: 10
SYNTH_NETS = ['SynthWide', 'SynthWide256', 'SynthWideAndLong']
# NUMBER OF VALID NETWORKS: 3
ARCH_SPACES = ['random_net_densenet', 'random_net_googlenet', 'random_net_mobilenetv2', 'random_net_pnasnet', 'random_net_probenet_s', 'random_net_resnet', 'random_net_resnext']
# NUMBER OF VALID ARCHSPACES: 7

ALL_NETS = KUANGLIU_NETS + PROBE_NETS + SYNTH_NETS + ARCH_SPACES


"""
The main entry routine to build the network.

:param netname: str, identifies the network
:param num_c: int, defines the amount of classes. Affects the last layer. Note, if you try to load latter on weights, their stored size MUST match the number of classes define here.
:param net_args: dict, defines network specific configurations, that is directly passed to the network constructor. 
:return: PyTorch network instance 
"""
def build(netname, num_c, net_args=None):
    if netname in KUANGLIU_NETS:
        net = getattr(kuangliu_models, netname)(num_c)

    elif netname in PROBE_NETS:
        net = getattr(probenets, netname)(num_c)

    elif netname in SYNTH_NETS:
        net = getattr(synthnets, netname)(num_c=num_c, **net_args)

    elif netname in ARCH_SPACES:
        net = getattr(iotnets, netname).get_instance(net_args, num_c)
        return net

    else:
        raise ValueError("netname = '{}' is not implemented or supported!".format(netname))

    return net


"""
Routines that help generating valid net_args.
Those routines are only valid for networks that are defined in the ARCH_SPACE constant. 

# SAMPLE TO BUILD A NETWORK DIRECTLY:
1: import diac_h2h.networks.pytorch.iotnets.random_net_resnext as random_net_resnext
2: config = random_net_resnext.get_config("s1")          # defines a sampling law
3: net_args = random_net_resnext.sample(config, 0)       # samples network id=0 from that law to generate specific 
     args.
4: net = random_net_resnext.get_instance(net_args, 12)   # constructs the network according the arguments


# HOW TO USE THE HELPER ROUTINES
1: law_config = get_law_config("random_net_densenet", "s1")
2: net_args = sample_from_law("random_net_densenet", law_config, 0)
3: net = build("random_net_densenet", 12, net_args)
"""

def get_law_config(netname, law):
    if netname in ARCH_SPACES:
        # net_args MUST match the search space requiements.
        net = getattr(iotnets, netname).get_config(law)
        return net
    else:
        raise ValueError("spacename {} not valid! Must be any of: {}".format(netname, ARCH_SPACES))


def sample_from_law(netname, law_config, id):
    if netname in ARCH_SPACES:
        # net_args MUST match the search space requiements.
        net = getattr(iotnets, netname).sample(law_config, id)
        return net
    else:
        raise ValueError("spacename {} not valid! Must be any of: {}".format(netname, ARCH_SPACES))
