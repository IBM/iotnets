import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.resnext import Block


def get_config(space):
    config = {}
    if space == "s1":
        config["num_blocks"] = [1, 3]
        config["bottleneck_widths"] = [4, 64]
        config["prod_limit"] = [512, 512+1]
    elif space == "s2":
        config["num_blocks"] = [1, 3]
        config["bottleneck_widths"] = [4, 32]
        config["prod_limit"] = [128, 128+1]
    elif space == "s3":
        config["num_blocks"] = [1, 2]
        config["bottleneck_widths"] = [4, 8]
        config["prod_limit"] = [32, 32+1]
    else:
        raise ValueError("Space {} not defined!".format(space))

    return config


def sample(config, id):
    rstate = np.random.RandomState(id)
    # num_blocks 1 - 4, bottleneck_widths 1-64, prod_limit 1 - 512
    num_blocks = list(rstate.randint(*config["num_blocks"], 3))  # original 3
    bottleneck_widths = list(rstate.randint(*config["bottleneck_widths"], 3))
    prod_limit = int(rstate.randint(*config["prod_limit"], 1)[0])

    cardinalities = [0, 0, 0]
    for i in range(3):
        clim = int(prod_limit / bottleneck_widths[i])
        clim = max(clim, 2)
        cardinalities[i] = int(rstate.randint(1, clim, 1)[0])

    return dict(num_blocks=num_blocks, cardinalities=cardinalities, bottleneck_widths=bottleneck_widths)


def get_instance(net_args, num_classes):
    return ResNeXt_modified(num_c=num_classes, **net_args)


class ResNeXt_modified(nn.Module):
    def __init__(self, num_blocks, cardinalities, bottleneck_widths, num_c=10):
        super(ResNeXt_modified, self).__init__()
        # ORIGNAL CONFIG ResNeXt29_2x64d
        # num_blocks = [3, 3, 3]
        # cardinalities = [2, 2, 2]
        # bottleneck_widths = [33, 66, 132]
        # ORIGNAL CONFIG ResNeXt29_32x4d
        # num_blocks = [3, 3, 3]
        # cardinalities = [32, 32, 32]
        # bottleneck_widths = [4, 8, 16]

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], int(cardinalities[0]), bottleneck_widths[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], int(cardinalities[1]), bottleneck_widths[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], int(cardinalities[2]), bottleneck_widths[2], 2)
        self.linear = nn.Linear(cardinalities[2]*bottleneck_widths[2]*2, num_c)

    def _make_layer(self, num_blocks, cardinality, bottleneck_width, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, cardinality, bottleneck_width, stride))
            self.in_planes = Block.expansion * cardinality * bottleneck_width
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
