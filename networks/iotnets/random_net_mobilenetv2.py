import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.mobilenetv2 import Block


def make_cfg(expansions, out_planes, num_blocks, strides):
    # assert all the same length
    ret = []
    for i in range(len(expansions)):
        e = int(expansions[i])
        o = int(out_planes[i])
        n = int(num_blocks[i])
        s = int(strides[i])
        entry = (e,o,n,s)
        ret.append(entry)
    return ret


def get_config(space):
    config = {}
    if space == "s1":
        config["expansions"] = [1, 8]
        config["out_planes"] = [16, 256]
        config["num_blocks"] = [1, 4]
        config["io_in"] = [16, 128]
        config["io_out"] = [128, 1280]
    elif space == "s2":
        config["expansions"] = [1, 6]
        config["out_planes"] = [16, 128]
        config["num_blocks"] = [1, 3]
        config["io_in"] = [16, 64]
        config["io_out"] = [128, 512]
    elif space == "s3":
        config["expansions"] = [1, 4]
        config["out_planes"] = [16, 64]
        config["num_blocks"] = [1, 2]
        config["io_in"] = [16, 32]
        config["io_out"] = [128, 256]
    elif space == "s4":
        config["expansions"] = [1, 4]
        config["out_planes"] = [4, 32]
        config["num_blocks"] = [1, 2]
        config["io_in"] = [16, 32]
        config["io_out"] = [64, 128]
    elif space == "s5":
        config["expansions"] = [1, 4]
        config["out_planes"] = [2, 8]
        config["num_blocks"] = [1, 2]
        config["io_in"] = [16, 32]
        config["io_out"] = [16, 64]
    elif space == "s6":
        config["expansions"] = [1, 2]
        config["out_planes"] = [2, 8]
        config["num_blocks"] = [1, 2]
        config["io_in"] = [4, 8]
        config["io_out"] = [12, 16]
    else:
        raise ValueError("Space {} not defined!".format(space))

    return config


def sample(config, id):
    rstate = np.random.RandomState(id)
    # expansions 1-8, out_planes 1-256, num_blocks 1-4, io_in 1-128, io_out 1-1280
    l = 7  # match the cfg length of the original
    expansions = rstate.randint(*config["expansions"], l)
    out_planes = rstate.randint(*config["out_planes"], l)
    num_blocks = rstate.randint(*config["num_blocks"], l)
    # strides array with all 1's except 3 times a 2.
    strides = np.ones(l, dtype=int)
    strides[0:3] = int(2)
    rstate.shuffle(strides)
    cfg = make_cfg(expansions, out_planes, num_blocks, strides)
    io_planes = [0, 0]
    io_planes[0] = int(rstate.randint(*config["io_in"], 1)[0])  # original value 32
    io_planes[1] = int(rstate.randint(*config["io_out"], 1)[0])  # original value 1280

    return dict(cfg=cfg, io_planes=io_planes)


def get_instance(net_args, num_classes):
    return MobileNetV2_modified(num_c=num_classes, **net_args)


class MobileNetV2_modified(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    # cfg = [(1,  16, 1, 1),
    #        (6,  24, 2, 1),
    #        (6,  32, 3, 2),
    #        (6,  64, 4, 2),
    #        (6,  96, 3, 1),
    #        (6, 160, 3, 2),
    #        (6, 320, 1, 1)]
    # important:
    # out_planes 320 needs to match downstream topology
    # strides: needs to be all 1's except 3 times a 2 to match spatial dimension reduction at the end.

    def __init__(self, cfg, io_planes, num_c=10):
        last_planes = cfg[-1][1]
        assert len(io_planes) == 2
        first_planes = io_planes[0]  # Original value: 32
        last_planes2 = io_planes[1]  # Original value: 1280

        super(MobileNetV2_modified, self).__init__()
        self.conv1 = nn.Conv2d(3, first_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(first_planes)
        self.layers = self._make_layers(cfg, in_planes=first_planes)
        self.conv2 = nn.Conv2d(last_planes, last_planes2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(last_planes2)
        self.linear = nn.Linear(last_planes2, num_c)

    def _make_layers(self, cfg, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
