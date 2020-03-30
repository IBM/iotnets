'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.resnet import BasicBlock
from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.resnet import Bottleneck


def get_config(space):
    config = {}
    if space == "s1":
        config["block_ops"] = [0, 1]
        config["fs"] = [16, 1024]
        config["num_blocks"] = [1, 4]
    elif space == "s2":
        config["block_ops"] = [0, 1]
        config["fs"] = [16, 512]
        config["num_blocks"] = [1, 3]
    elif space == "s3":
        config["block_ops"] = [0, 1]
        config["fs"] = [16, 128]
        config["num_blocks"] = [1, 3]
    elif space == "s4":
        config["block_ops"] = [0, 1]
        config["fs"] = [4, 32]
        config["num_blocks"] = [1, 3]
    elif space == "s5":
        config["block_ops"] = [0, 1]
        config["fs"] = [2, 8]
        config["num_blocks"] = [1, 3]
    else:
        raise ValueError("Space {} not defined!".format(space))

    return config


def sample(config, id):
    rstate = np.random.RandomState(id)
    blocks = rstate.choice(config["block_ops"], 4)
    fs = rstate.randint(*config["fs"], 5)
    num_blocks = rstate.randint(*config["num_blocks"], 4)

    return dict(blocks=blocks, fs=fs, num_blocks=num_blocks)


def get_instance(net_args, num_classes):
    return ResNet_modified(num_c=num_classes, **net_args)


class ResNet_modified(nn.Module):
    def __init__(self, blocks, fs, num_blocks, num_c=10):
        super(ResNet_modified, self).__init__()
        assert len(fs) == 5
        assert len(num_blocks) == 4
        assert len(blocks) == 4
        blocks = list(map(self._int2block, blocks))

        self.in_planes = fs[0]
        self.conv1 = nn.Conv2d(3, fs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(fs[0])
        self.layer1 = self._make_layer(blocks[0], fs[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], fs[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], fs[3], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(blocks[3], fs[4], num_blocks[3], stride=2)
        self.linear = nn.Linear(fs[4]*blocks[3].expansion, num_c)

    def _int2block(self, block):
        if block == 0:
            return BasicBlock
        elif block == 1:
            return Bottleneck
        else:
            raise ValueError("block code {} not supported".format(block))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
