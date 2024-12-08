import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# stage_out_channel = [64] * 2 + [128] * 2 + [256] * 2

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


# def binaryconv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def channel_wise_pool(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False, groups=outplanes)
    

# class channel_wise_pool(nn.Module):
#     def __init__(self, inplanes, outplanes):
#         super(channel_wise_pool, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False, groups=outplanes)

#     def forward(self, x):
#         out = self.conv1(x)
#         return out


class firstconv(nn.Module):
    def __init__(self, inp, oup):
        super(firstconv, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup//4, 3, 1, 1, bias=False)
        self.px1 = nn.PixelUnshuffle(2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.px1(out)

        return out
    

class lastconv(nn.Module):
    def __init__(self, inp, oup):
        super(lastconv, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup*4, 3, 1, 1, bias=False)
        self.px1 = nn.PixelShuffle(2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.px1(out)

        return out


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LearnableScale(nn.Module):
    def __init__(self, out_chn):
        super(LearnableScale, self).__init__()
        self.scale = nn.Parameter(torch.ones(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x * self.scale.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.move11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = conv3x3(inplanes, inplanes//2, stride)
        self.channel_pool = channel_wise_pool(inplanes, inplanes//2)
        self.px = nn.PixelUnshuffle(2)

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        self.planes = planes

    def forward(self, x):

        out = self.move11(x)
        out = self.prelu1(out)
        out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)

        shortcut = self.channel_pool(x)

        out = out + shortcut

        out = self.px(out)

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
        

class BasicBlock_transpose(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_transpose, self).__init__()

        self.move11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = conv3x3(inplanes, inplanes*2, stride)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale(inplanes)

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        self.planes = planes


    def forward(self, x):
        out = self.move11(x)
        out = self.prelu1(out)
        out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)

        shortcut = self.scale(x)
        shortcut = torch.cat((shortcut, shortcut), 1)

        out = out + shortcut

        out = self.px(out)

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out


class BinaryBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BinaryBlock, self).__init__()

        self.move11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = binaryconv3x3(inplanes, inplanes//2, stride)
        self.channel_pool = channel_wise_pool(inplanes, inplanes//2)
        self.px = nn.PixelUnshuffle(2)

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        self.planes = planes

    def forward(self, x):

        out = self.move11(x)
        out = self.prelu1(out)
        out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)

        shortcut = self.channel_pool(x)

        out = out + shortcut

        out = self.px(out)

        assert out.shape[1] == self.planes

        return out


class BinaryBlock_transpose(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BinaryBlock_transpose, self).__init__()

        self.move11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = binaryconv3x3(inplanes, inplanes*2, stride)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale(inplanes)

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        self.planes = planes

    def forward(self, x):
        out = self.move11(x)
        out = self.prelu1(out)
        out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)

        shortcut = self.scale(x)
        shortcut = torch.cat((shortcut, shortcut), 1)

        out = out + shortcut

        out = self.px(out)

        assert out.shape[1] == self.planes

        return out


class BDJSCC(nn.Module):
    def __init__(self, init_channels=32):
        super(BDJSCC, self).__init__()
        self.firstconv = firstconv(3, init_channels)
        self.layer1 = BasicBlock(init_channels, init_channels*2)
        self.layer2 = BasicBlock(init_channels*2, init_channels*4)
        self.layer3 = BasicBlock(init_channels*4, init_channels*8)
        self.layer4 = BasicBlock_transpose(init_channels*8, init_channels*4)
        self.layer5 = BasicBlock_transpose(init_channels*4, init_channels*2)
        self.layer6 = BasicBlock_transpose(init_channels*2, init_channels)
        self.lastconv = lastconv(init_channels, 3)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        out = self.firstconv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.lastconv(out)
        out = self.final_activation(out)

        return out
    

class BDJSCC_Binary(nn.Module):
    def __init__(self, init_channels=32):
        super(BDJSCC_Binary, self).__init__()
        self.firstconv = firstconv(3, init_channels)
        self.layer1 = BinaryBlock(init_channels, init_channels*2)
        self.layer2 = BinaryBlock(init_channels*2, init_channels*4)
        self.layer3 = BinaryBlock(init_channels*4, init_channels*8)
        self.layer4 = BinaryBlock_transpose(init_channels*8, init_channels*4)
        self.layer5 = BinaryBlock_transpose(init_channels*4, init_channels*2)
        self.layer6 = BinaryBlock_transpose(init_channels*2, init_channels)
        self.lastconv = lastconv(init_channels, 3)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        out = self.firstconv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.lastconv(out)
        out = self.final_activation(out)

        return out

    