import torch
import torch.nn as nn
from ABNet import *
from channel import Channel


def Bconv3x3(in_planes, out_planes, stride=1):
    """3x3 binary convolution with padding"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def conv3x3(in_planes, out_planes, stride=1):
    return ScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x, beta=1):
        out = x + self.bias.expand_as(x) / beta
        return out


class firstconv(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(firstconv, self).__init__()

        self.conv1 = ScaledStdConv2d(inp, oup//4, 3, stride, 1, bias=False, use_layernorm=True)
        self.px1 = nn.PixelUnshuffle(2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.px1(out)

        return out


class lastconv(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(lastconv, self).__init__()

        self.conv1 = ScaledStdConv2d(inp, oup*4, 3, stride, 1, bias=False, use_layernorm=True)
        self.px1 = nn.PixelShuffle(2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.px1(out)

        return out



class ABBlock_thick(nn.Module):
    def __init__(self, inplanes, planes, alpha, stride=1, binary=False, expected_var=1.0):
        super(ABBlock_thick, self).__init__()

        self.alpha = alpha
        self.beta1 = 1. / expected_var ** 0.5
        expected_var += alpha ** 2
        self.beta2 = 1. / expected_var ** 0.5

        self.move11 = LearnableBias(inplanes)
        if binary:
            self.conv1 = Bconv3x3(inplanes, inplanes, stride=stride)
        else:
            self.conv1 = conv3x3(inplanes, inplanes, stride=stride)
        # self.maxout1 = Maxout(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.prelu1 = Q_PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)
        if binary:
            self.conv2 = Bconv3x3(inplanes, inplanes//2, stride=stride)
        else:
            self.conv2 = conv3x3(inplanes, inplanes//2, stride=stride)
        # self.maxout2 = Maxout(inplanes//2)
        self.move22 = LearnableBias(inplanes//2)
        self.prelu2 = Q_PReLU(inplanes//2)
        self.move23 = LearnableBias(inplanes//2)

        self.px = nn.PixelUnshuffle(2)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes

        self.planes = planes

    def forward(self, x):
        
        out = self.move11(x, beta=self.beta1)
        out = self.binary_activation(out*self.beta1)
        out = self.conv1(out)
        # out = self.maxout1(out)
        out = self.move12(out)
        out = self.prelu1(out)
        out = self.move13(out)

        # shortcut = self.scale1(x)
        x = out*self.alpha + x

        out = self.move21(x, beta=self.beta2)
        out = self.binary_activation(out*self.beta2)
        out = self.conv2(out)
        # out = self.maxout2(out)
        out = self.move22(out)
        out = self.prelu2(out)
        out = self.move23(out)

        shortcut = x[:, :x.shape[1]//2, :, :]
        # shortcut = self.scale2(shortcut)

        out = out*self.alpha + shortcut

        out = self.px(out)

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
    
class ABBlock_thick_transpose(nn.Module):
    def __init__(self, inplanes, planes, alpha, stride=1, binary=False, expected_var=1.0):
        super(ABBlock_thick_transpose, self).__init__()

        self.alpha = alpha
        self.beta1 = 1. / expected_var ** 0.5
        expected_var += alpha ** 2
        self.beta2 = 1. / expected_var ** 0.5
        

        self.move11 = LearnableBias(inplanes)
        if binary:
            self.conv1 = Bconv3x3(inplanes, inplanes, stride=stride)
        else:
            self.conv1 = conv3x3(inplanes, inplanes, stride=stride)
        # self.maxout1 = Maxout(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.prelu1 = Q_PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)
        if binary:
            self.conv2 = Bconv3x3(inplanes, inplanes*2, stride=stride)
        else:
            self.conv2 = conv3x3(inplanes, inplanes*2, stride=stride)
        # self.maxout2 = Maxout(inplanes*2)
        self.move22 = LearnableBias(inplanes*2)
        self.prelu2 = Q_PReLU(inplanes*2)
        self.move23 = LearnableBias(inplanes*2)

        self.px = nn.PixelShuffle(2)


        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes

        self.planes = planes

    def forward(self, x):
        
        out = self.move11(x, beta=self.beta1)
        out = self.binary_activation(out*self.beta1)
        out = self.conv1(out)
        # out = self.maxout1(out)
        out = self.move12(out)
        out = self.prelu1(out)
        out = self.move13(out)

        # shortcut = self.scale1(x)
        x = out*self.alpha + x

        out = self.move21(x, beta=self.beta2)
        out = self.binary_activation(out*self.beta2)
        out = self.conv2(out)
        # out = self.maxout2(out)
        out = self.move22(out)
        out = self.prelu2(out)
        out = self.move23(out)

        shortcut = torch.cat((x, x), 1)
        # shortcut = self.scale2(shortcut)

        out = out*self.alpha + shortcut

        out = self.px(out)

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out


class BDJSCC_AB(nn.Module):
    def __init__(self, init_channels=64, channel_type='awgn', snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None, alpha=0.25, binary=True):
        super(BDJSCC_AB, self).__init__()
        self.firstconv = firstconv(3, init_channels)
        expected_var = 1.0
        self.layer1 = ABBlock_thick(init_channels, init_channels*2, alpha, expected_var=expected_var, binary=binary)
        expected_var += 2 * alpha ** 2
        self.layer2 = ABBlock_thick(init_channels*2, init_channels*4, alpha, expected_var=expected_var, binary=binary)
        expected_var += 2 * alpha ** 2
        self.layer3 = ABBlock_thick(init_channels*4, init_channels*8, alpha, expected_var=expected_var, binary=binary)
        expected_var += 2 * alpha ** 2
        self.layer3_ = ABBlock_thick(init_channels*8, init_channels*16, alpha, expected_var=expected_var, binary=binary)
        expected_var += 2 * alpha ** 2
        self.channel = Channel(channel_type=channel_type)
        self.layer4_ = ABBlock_thick_transpose(init_channels*16, init_channels*8, alpha, expected_var=expected_var, binary=binary)
        expected_var += 2 * alpha ** 2
        self.layer4 = ABBlock_thick_transpose(init_channels*8, init_channels*4, alpha, expected_var=expected_var, binary=binary)
        expected_var += 2 * alpha ** 2
        self.layer5 = ABBlock_thick_transpose(init_channels*4, init_channels*2, alpha, expected_var=expected_var, binary=binary)
        expected_var += 2 * alpha ** 2
        self.layer6 = ABBlock_thick_transpose(init_channels*2, init_channels, alpha, expected_var=expected_var, binary=binary)
        self.lastconv = lastconv(init_channels, 3)
        self.final_activation = nn.Sigmoid()

        self.snr_db = snr_db
        self.h_real = h_real
        self.h_imag = h_imag
        self.b_prob = b_prob
        self.b_stddev = b_stddev

    def forward(self, x):
        out = self.firstconv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer3_(out)
        out = self.channel(out, self.snr_db, self.h_real, self.h_imag, self.b_prob, self.b_stddev)
        out = self.layer4_(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.lastconv(out)
        out = self.final_activation(out)

        return out   