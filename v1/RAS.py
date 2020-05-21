import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet50 import ResNet50
import torchvision.models as models

class MSCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSCM, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4), nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.ReLU(True),
        )
        self.score = nn.Conv2d(out_channel*4, 1, 3, padding=1)

    def forward(self, x):
        x = self.convert(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.score(x)

        return x

class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):
        a = torch.sigmoid(-y)
        x = self.convert(x)
        x = a.expand(-1, self.channel, -1, -1).mul(x)
        y = y + self.convs(x)

        return y

class RAS(nn.Module):
    def __init__(self, channel=64):
        super(RAS, self).__init__()
        self.resnet = ResNet50()
        self.mscm = MSCM(2048, channel)
        self.ra1 = RA(64, channel)
        self.ra2 = RA(256, channel)
        self.ra3 = RA(512, channel)
        self.ra4 = RA(1024, channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.initialize_weights()

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.resnet(x)
        x_size = x.size()[2:]
        x1_size = x1.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        y5 = self.mscm(x5)
        score5 = F.interpolate(y5, x_size, mode='bilinear', align_corners=True)

        y5_4 = F.interpolate(y5, x4_size, mode='bilinear', align_corners=True)
        y4 = self.ra4(x4, y5_4)
        score4 = F.interpolate(y4, x_size, mode='bilinear', align_corners=True)

        y4_3 = F.interpolate(y4, x3_size, mode='bilinear', align_corners=True)
        y3 = self.ra3(x3, y4_3)
        score3 = F.interpolate(y3, x_size, mode='bilinear', align_corners=True)

        y3_2 = F.interpolate(y3, x2_size, mode='bilinear', align_corners=True)	
        y2 = self.ra2(x2, y3_2)
        score2 = F.interpolate(y2, x_size, mode='bilinear', align_corners=True)

        y2_1 = F.interpolate(y2, x1_size, mode='bilinear', align_corners=True)
        y1 = self.ra1(x1, y2_1)
        score1 = F.interpolate(y1, x_size, mode='bilinear', align_corners=True)

        return score1, score2, score3, score4, score5

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        self.resnet.load_state_dict(res50.state_dict(), False)
