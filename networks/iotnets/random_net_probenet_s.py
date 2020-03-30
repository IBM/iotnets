import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_config(space):
    config = {}
    if space == "s1":  # regular
        config["fs"] = [4, 32]
        config["kops"] = [1,3,5,7]
    elif space == "s2": # small networks
        config["fs"] = [2, 8]
        config["kops"] = [1,3,5]
    elif space == "s3": # larger networks
        config["fs"] = [16, 128]
        config["kops"] = [1,3,5,7]
    elif space == "s0":  # FULL SPACE
        config["fs"] = [1, 128]
        config["kops"] = [1,3,5,7]
    else:
        raise ValueError("Space {} not defined!".format(space))

    return config


def sample(config, id):
    rstate = np.random.RandomState(id)
    fs = rstate.randint(*config["fs"], 3)
    ks = rstate.choice(config["kops"], 6)

    return dict(fs=fs, ks=ks)


def get_instance(net_args,num_classes):
    return ProbeNet_s_modified(num_c=num_classes, **net_args)


def pad_same(k):
    # OUT: n + 2p - k + 1
    # OUT as IN: 2p - k + 1 =!= 0.
    p = int((k-1)/2)
    return p


class ProbeNet_s_modified(nn.Module):
    def __init__(self, num_c, fs, ks):
        # Default values for ProbeNet_s.
        # f1 = 8 # out of 4 - 32
        # f2 = 16 # out of 4 - 32
        # f3 = 32 # out of 4 - 32
        # ks array of 6 entries in: 1,3,5,7
        assert len(ks) == 6
        assert len(fs) == 3
        f1 = fs[0]
        f2 = fs[1]
        f3 = fs[2]

        # d = 512 # input 2^3 downsample => 4*4*f3.
        self.densedim = 4*4*f3
        super(ProbeNet_s_modified, self).__init__()
        self.conv0 = nn.Conv2d(3, f1, [ks[0], ks[1]], 1, [pad_same(ks[0]), pad_same(ks[1])], bias=False)
        self.bn_c0 = nn.BatchNorm2d(f1)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1 = nn.Conv2d(f1, f2, [ks[2], ks[3]], 1, [pad_same(ks[2]), pad_same(ks[3])], bias=False)
        self.bn_c1 = nn.BatchNorm2d(f2)
        self.pool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv2 = nn.Conv2d(f2, f3, [ks[4], ks[5]], 1, [pad_same(ks[4]), pad_same(ks[5])], bias=False)
        self.bn_c2 = nn.BatchNorm2d(f3)
        self.pool3 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(self.densedim, num_c)


    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(-1, self.densedim)
        x = self.fc_out(x)
        return x
