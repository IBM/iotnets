'''GoogLeNet with PyTorch.'''

"SEE: https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7"
"SOURCE: Going Deeper with Convolutions"

import torch
import torch.nn as nn
import numpy as np

from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.googlenet import Inception


def get_config(space):
    config = {}
    if space == "s1":
        config["io_in"] = [16, 256]
        config["l1"] = [16, 384]
        config["l2"] = [16, 192]
        config["l3"] = [16, 384]
        config["l4"] = [16, 48]
        config["l5"] = [16, 128]
        config["l6"] = [16, 128]
    elif space == "s2":
        config["io_in"] = [16, 256]
        config["l1"] = [16, 64]
        config["l2"] = [16, 64]
        config["l3"] = [16, 64]
        config["l4"] = [16, 32]
        config["l5"] = [16, 32]
        config["l6"] = [16, 32]
    elif space == "s3":
        config["io_in"] = [16, 256]
        config["l1"] = [16, 32]
        config["l2"] = [8, 16]
        config["l3"] = [16, 32]
        config["l4"] = [4, 8]
        config["l5"] = [4, 8]
        config["l6"] = [4, 8]
    elif space == "s4":
        config["io_in"] = [16, 128]
        config["l1"] = [4, 8]
        config["l2"] = [4, 8]
        config["l3"] = [4, 8]
        config["l4"] = [4, 8]
        config["l5"] = [4, 8]
        config["l6"] = [4, 8]
    elif space == "s5":
        config["io_in"] = [8, 16]
        config["l1"] = [4, 6]
        config["l2"] = [4, 6]
        config["l3"] = [4, 6]
        config["l4"] = [4, 6]
        config["l5"] = [4, 6]
        config["l6"] = [4, 6]
    elif space == "s6":
        config["io_in"] = [8, 16]
        config["l1"] = [2, 6]
        config["l2"] = [2, 6]
        config["l3"] = [2, 6]
        config["l4"] = [2, 6]
        config["l5"] = [2, 6]
        config["l6"] = [2, 6]
    else:
        raise ValueError("Space {} not defined!".format(space))

    return config


def sample(config, id):
    rstate = np.random.RandomState(id)
    # io_in 1 - 256, l1, l2 ... l6 each in 1 - 512
    l = 9  # match the cfg length of the original
    init_planes = int(rstate.randint(*config["io_in"], 1)[0])  # original value 192
    cfg = []
    for i in range(l):
        # this code ensures backwards compatibility with old code.
        # The line has no logical effect, except that it changes the seeds in different ways
        if config["l1"][1] > 60:
            # s1 and s2
            line = rstate.randint(16, 512, 7)
        else:
            # s3, s4, s5, s6.
            line = rstate.randint(16, 256, 7)
        line = np.zeros(7, dtype=int)  # 7 length of parms
        line[0] = init_planes
        line[1] = int(rstate.randint(*config["l1"], 1)[0])
        line[2] = int(rstate.randint(*config["l2"], 1)[0])
        line[3] = int(rstate.randint(*config["l3"], 1)[0])
        line[4] = int(rstate.randint(*config["l4"], 1)[0])
        line[5] = int(rstate.randint(*config["l5"], 1)[0])
        line[6] = int(rstate.randint(*config["l6"], 1)[0])
        init_planes = line[1] + line[3] + line[5] + line[6]  # Inception: cat-module need to match next block!
        cfg.append(line)
    return cfg


def get_instance(cfg,num_classes):
    return GoogLeNet_modified(cfg, num_c=num_classes)


# ORIGINAL
# self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
# self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
# self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
# self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
# self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
# self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
# self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
# self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
# self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

class GoogLeNet_modified(nn.Module):
    def __init__(self, cfg, num_c=10):
        super(GoogLeNet_modified, self).__init__()

        assert len(cfg) == 9
        first_planes = cfg[0][0]
        last_planes = cfg[-1][1] + cfg[-1][3] + cfg[-1][5] + cfg[-1][6]

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, first_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(first_planes),
            nn.ReLU(True),
        )

        self.a3 = Inception(*cfg[0])
        self.b3 = Inception(*cfg[1])

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(*cfg[2])
        self.b4 = Inception(*cfg[3])
        self.c4 = Inception(*cfg[4])
        self.d4 = Inception(*cfg[5])
        self.e4 = Inception(*cfg[6])

        self.a5 = Inception(*cfg[7])
        self.b5 = Inception(*cfg[8])

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(last_planes, num_c)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
