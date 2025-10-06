import torch
import torch.nn as nn
import torch.nn.functional as F

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class DAM(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.ch_att = _ChannelAttentionModule()
        self.s_att = _PositionAttentionModule(in_channels)
    def forward(self, inputs):
        x = self.ch_att(inputs)
        y = self.s_att(inputs)
        out = x + y
        return out




class build_model(nn.Module):
    def __init__(self, in_c, out_c, rate=[3, 6, 9]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True), 
            Squeeze_Excitation(out_c//5)
            
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True), 
            Squeeze_Excitation(out_c//5)

        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True), 
            Squeeze_Excitation(out_c//5)

        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c//5, kernel_size=1),
            nn.BatchNorm2d(out_c//5),
            nn.ReLU(inplace=True), 
            Squeeze_Excitation(out_c//5)


        )



        self.c5 = nn.Conv2d(in_c, out_c//5, kernel_size=1, padding=0)
        self.dual_att = DAM(out_c//5)
        self.conv1 = nn.Conv2d((out_c//5)*5, out_c, kernel_size=1)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x5 = self.c5(inputs)
        x5 = self.dual_att(x5)
        x = torch.cat((x1, x2, x3, x4, x5), axis=1)
        x = self.conv1(x)
        return x
