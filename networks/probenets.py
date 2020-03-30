import torch.nn as nn
import torch.nn.functional as F
import numpy as np

####################################################################################
### CONTAINS ALL PROBENETS ###
####################################################################################
### SEE: https://www.researchgate.net/publication/324055964_Efficient_Image_Dataset_Classification_Difficulty_Estimation_for_Predicting_Deep-Learning_Accuracy
### Page 4, Figure 2 a) - f)
####################################################################################
# Figure a) regular, narrow and wide
####################################################################################
class ProbeNet_s(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_s, self).__init__()
        self.conv0 = nn.Conv2d(3, 8, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1 = nn.Conv2d(8, 16, [3, 3], [1, 1], 1, bias=False)
        self.bn_c1 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv2 = nn.Conv2d(16, 32, [3, 3], [1, 1], 1, bias=False)
        self.bn_c2 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(512, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc_out(x)
        return x


class ProbeNet_s_slim(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_s_slim, self).__init__()
        self.conv0 = nn.Conv2d(3, 2, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(2)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1 = nn.Conv2d(2, 4, [3, 3], [1, 1], 1, bias=False)
        self.bn_c1 = nn.BatchNorm2d(4)
        self.pool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv2 = nn.Conv2d(4, 8, [3, 3], [1, 1], 1, bias=False)
        self.bn_c2 = nn.BatchNorm2d(8)
        self.pool3 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(128, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(-1, 8 * 4 * 4)
        x = self.fc_out(x)
        return x


class ProbeNet_s_fat(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_s_fat, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1 = nn.Conv2d(32, 64, [3, 3], [1, 1], 1, bias=False)
        self.bn_c1 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv2 = nn.Conv2d(64, 128, [3, 3], [1, 1], 1, bias=False)
        self.bn_c2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(2048, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc_out(x)
        return x


####################################################################################
# Figure b) shallow
####################################################################################
class ProbeNet_s_shallow(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_s_shallow, self).__init__()
        self.conv0 = nn.Conv2d(3, 8, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(2048, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = x.view(-1, 8 * 16 * 16)
        x = self.fc_out(x)
        return x


class ProbeNet_s_nr_shallow(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_s_nr_shallow, self).__init__()
        self.conv0 = nn.Conv2d(3, 2, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(2)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(512, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = x.view(-1, 2 * 16 * 16)
        x = self.fc_out(x)
        return x


####################################################################################
# Figure c) deep
####################################################################################
class ProbeNet_s_deep(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_s_deep, self).__init__()
        self.conv0 = nn.Conv2d(3, 8, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1 = nn.Conv2d(8, 16, [3, 3], [1, 1], 1, bias=False)
        self.bn_c1 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv2 = nn.Conv2d(16, 32, [3, 3], [1, 1], 1, bias=False)
        self.bn_c2 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv3 = nn.Conv2d(32, 64, [3, 3], [1, 1], 1, bias=False)
        self.bn_c3 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv4 = nn.Conv2d(64, 128, [3, 3], [1, 1], 1, bias=False)
        self.bn_c4 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(128, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = self.pool3(x)
        x = F.relu(self.bn_c3(self.conv3(x)))
        x = self.pool4(x)
        x = F.relu(self.bn_c4(self.conv4(x)))
        x = self.pool5(x)
        x = x.view(-1, 128 * 1 * 1)
        x = self.fc_out(x)
        return x


class ProbeNet_s_nr_deep(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_s_nr_deep, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1 = nn.Conv2d(32, 64, [3, 3], [1, 1], 1, bias=False)
        self.bn_c1 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv2 = nn.Conv2d(64, 128, [3, 3], [1, 1], 1, bias=False)
        self.bn_c2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv3 = nn.Conv2d(128, 256, [3, 3], [1, 1], 1, bias=False)
        self.bn_c3 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv4 = nn.Conv2d(256, 512, [3, 3], [1, 1], 1, bias=False)
        self.bn_c4 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc_out = nn.Linear(512, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = self.pool3(x)
        x = F.relu(self.bn_c3(self.conv3(x)))
        x = self.pool4(x)
        x = F.relu(self.bn_c4(self.conv4(x)))
        x = self.pool5(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc_out(x)
        return x


####################################################################################
# Figure d), e) and f)
####################################################################################
class ProbeNet_d_mlp(nn.Module):
    def __init__(self, num_c):
        n1 = round(32 * 32 * .75 + .25 * num_c)
        n2 = round(32 * 32 * .5 + .5 * num_c)
        n3 = round(32 * 32 * .25 + .75 * num_c)

        super(ProbeNet_d_mlp, self).__init__()
        self.fc0 = nn.Linear(3072, n1, bias=False)
        self.bn_dense0 = nn.BatchNorm1d(n1)
        self.fc1 = nn.Linear(n1, n2, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(n2)
        self.fc2 = nn.Linear(n2, n3, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(n3)
        self.fc_out = nn.Linear(n3, num_c)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.bn_dense0(self.fc0(x)))
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.fc_out(x)
        return x


class ProbeNet_d_feature(nn.Module):
    def __init__(self, num_c):
        f1 = int(round(8 * np.log10(num_c)))
        f2 = int(round(4 * np.sqrt(num_c)))
        f3 = num_c
        self.num_c = num_c

        super(ProbeNet_d_feature, self).__init__()
        self.conv0 = nn.Conv2d(3, f1, [3, 3], [1, 1], 1, bias=False)
        self.bn_c0 = nn.BatchNorm2d(f1)
        self.pool1 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1 = nn.Conv2d(f1, f2, [3, 3], [1, 1], 1, bias=False)
        self.bn_c1 = nn.BatchNorm2d(f2)
        self.pool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.conv2 = nn.Conv2d(f2, f3, [3, 3], [1, 1], 1, bias=False)
        self.bn_c2 = nn.BatchNorm2d(f3)
        self.pool3 = nn.MaxPool2d([2, 2], [2, 2])
        self.fc0 = nn.Linear(4 * 4 * f3, 2 * num_c, bias=False)
        self.bn_dense0 = nn.BatchNorm1d(2 * num_c)
        self.fc_out = nn.Linear(2 * num_c, num_c)

    def forward(self, x):
        x = F.relu(self.bn_c0(self.conv0(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_c1(self.conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_c2(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(-1, self.num_c * 4 * 4)
        x = F.relu(self.bn_dense0(self.fc0(x)))
        x = self.fc_out(x)
        return x


class Chained_Convs(nn.Module):
    def __init__(self, f_in, f, r):
        super(Chained_Convs, self).__init__()
        self.conv_in = nn.Conv2d(f_in, f, [3, 3], [1, 1], 1, bias=False)
        self.bn = nn.BatchNorm2d(f)
        self.conv_rep = nn.Conv2d(f, f, [3, 3], [1, 1], 1, bias=False)
        self.pool = nn.MaxPool2d([2, 2], [2, 2])
        self.rep = r - 1

    def forward(self, x):
        x = F.relu(self.bn(self.conv_in(x)))
        for r in range(self.rep):
            x = F.relu(self.bn(self.conv_rep(x)))
        x = self.pool(x)
        return x


class ProbeNet_d_length(nn.Module):
    def __init__(self, num_c):
        super(ProbeNet_d_length, self).__init__()

        rep = int(round(np.log10(num_c)))
        num_middle = int(round((num_c + 2 * 2 * 64) / 2))

        self.chain1 = Chained_Convs(3, 8, rep)
        self.chain2 = Chained_Convs(8, 16, rep)
        self.chain3 = Chained_Convs(16, 32, rep)
        self.chain4 = Chained_Convs(32, 64, rep)
        self.chain5 = Chained_Convs(64, 128, rep)
        self.fc0 = nn.Linear(128, num_middle, bias=False)
        self.bn_dense0 = nn.BatchNorm1d(num_middle)
        self.fc_out = nn.Linear(num_middle, num_c)

    def forward(self, x):
        x = self.chain1(x)
        x = self.chain2(x)
        x = self.chain3(x)
        x = self.chain4(x)
        x = self.chain5(x)
        x = x.view(-1, 128 * 1 * 1)
        x = F.relu(self.bn_dense0(self.fc0(x)))
        x = self.fc_out(x)
        return x
