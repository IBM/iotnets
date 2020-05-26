import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.pnasnet import CellA
from diac_h2h.networks.pytorch.thirdparty.kuangliu_models.pnasnet import CellB


def get_config(space):
    config = {}
    if space == "s1":
        config["num_cells"] = [1, 12]
        config["num_planes0"] = [16, 128]
        config["num_planes1"] = [1, 4]
        config["num_planes2"] = [1, 4]
    elif space == "s2":
        config["num_cells"] = [1, 8]
        config["num_planes0"] = [16, 64]
        config["num_planes1"] = [1, 3]
        config["num_planes2"] = [1, 3]
    elif space == "s3":
        config["num_cells"] = [1, 3]
        config["num_planes0"] = [8, 16]
        config["num_planes1"] = [1, 2]
        config["num_planes2"] = [1, 2]
    else:
        raise ValueError("Space {} not defined!".format(space))

    return config


def sample(config, id):
    rstate = np.random.RandomState(id)
    # num_cells 1 - 12 num_planes0 1 - 128, num_planes1 1-4, num_planes2 1-4
    cell_types = [0]  # all A cells
    num_cells = rstate.randint(*config["num_cells"], 3)  # original 6
    # original 44, x*2 and x*4 (division with no remainder -> group pram!)
    num_planes = [0, 0, 0]
    num_planes[0] = int(rstate.randint(*config["num_planes0"], 1)[0])
    num_planes[1] = num_planes[0] * int(rstate.randint(*config["num_planes1"], 1)[0])
    num_planes[2] = num_planes[1] * int(rstate.randint(*config["num_planes2"], 1)[0])

    return dict(cell_types=cell_types, num_cells=num_cells, num_planes=num_planes)


def get_instance(net_args, num_classes):
    return PNASNet_modified(num_c=num_classes, **net_args)


class PNASNet_modified(nn.Module):
    def __init__(self, cell_types, num_cells, num_planes, num_c=10):
        super(PNASNet_modified, self).__init__()

        assert len(num_cells) == 3
        assert len(num_planes) == 3

        if len(cell_types) == 1:
            z = cell_types[0]
            # use for all the same
            cell_types = [0,0,0]
            for i in range(3):
                cell_types[i] = [z]*num_cells[i]

            cell_types_down = [z,z]

        self.in_planes = num_planes[0]

        self.conv1 = nn.Conv2d(3, num_planes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes[0])

        self.layer1 = self._make_layer(num_planes[0], cell_types[0], num_cells=num_cells[0])
        self.layer2 = self._downsample(num_planes[1], cell_types_down[0])
        self.layer3 = self._make_layer(num_planes[1], cell_types[1], num_cells=num_cells[1])
        self.layer4 = self._downsample(num_planes[2], cell_types_down[1])
        self.layer5 = self._make_layer(num_planes[2], cell_types[2], num_cells=num_cells[2])

        self.linear = nn.Linear(num_planes[2], num_c)

    def _int2cell(self, x):
        if x == 0:
            return CellA
        elif x == 1:
            return CellB
        else:
            raise ValueError("block code {} not supported".format(block))

    def _make_layer(self, planes, cell_types, num_cells):
        layers = []
        for i in range(num_cells):
            cell_type = self._int2cell(cell_types[i])
            layers.append(cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes, cell_type):
        cell_type = self._int2cell(cell_type)
        layer = cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 8)
        out = self.linear(out.view(out.size(0), -1))
        return out
