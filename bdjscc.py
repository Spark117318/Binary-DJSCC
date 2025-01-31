import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import PIL 
import os
from adalib import BinaryActivation as BA_ada
from adalib import AdaBin_Conv2d as conv_ada
from adalib import Maxout
from channel import Channel
from SEblock import AFlayer

# stage_out_channel = [64] * 2 + [128] * 2 + [256] * 2

def store_internal_output(output, filename):
    output = output[0]
    output = output.unsqueeze(1)
    print(output.shape)

    grid = vutils.make_grid(output, nrow=round(output.shape[0]**0.5), normalize=True)
    print(grid.shape)

    # Convert to PIL image
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    print(ndarr.shape)
    im = PIL.Image.fromarray(ndarr)

    images_path = os.path.join(os.getcwd(), 'internal_output')
    os.makedirs(images_path, exist_ok=True)

    im.save(os.path.join(images_path, filename))



def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

class firstconv_1(nn.Module):
    def __init__(self, inp, oup):
        super(firstconv_1, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, 2, 1, bias=False)
        # self.px1 = nn.PixelUnshuffle(2)

    def forward(self, x):

        out = self.conv1(x)
        # out = self.px1(out)

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

class lastconv_2(nn.Module):
    def __init__(self, inp, oup):
        super(lastconv_2, self).__init__()

        self.px1 = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(inp//4, oup, 3, 1, 1, bias=False)

    def forward(self, x):

        out = self.px1(x)
        out = self.conv1(out)

        return out


class lastconv_1(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(lastconv_1, self).__init__()

        self.move11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.conv2 = conv3x3(inplanes//4, planes, stride)
        # self.conv1x1 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        self.planes = planes


    def forward(self, x):

        out = self.binary_activation(x)
        out = self.conv1(out)
        out = self.move11(out)
        out = self.prelu1(out)
        out = self.move12(out)

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        # shortcut = self.scale(x)
        # shortcut = torch.cat((shortcut, shortcut), 1)
        # shortcut = self.conv1x1(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + x

        out = self.px(out)

        out = self.conv2(out)

        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out


class lastconv_modified(nn.Module):
    def __init__(self, inp, oup):
        super(lastconv_modified, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, 1, 1)

    def forward(self, x):

        out = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.conv1(out)

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
    def __init__(self):
        super(LearnableScale, self).__init__()
        self.scale = nn.Parameter(torch.ones(1,1,1,1), requires_grad=True)

    def forward(self, x):
        out = x * self.scale.expand_as(x)
        return out

class LearnableScale_per_channel(nn.Module):
    def __init__(self, out_chn):
        super(LearnableScale_per_channel, self).__init__()
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
        # print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -10.0, 10.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # print(binary_weights.mean())
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
        # self.channel_pool = channel_wise_pool(inplanes, inplanes//2)
        self.px = nn.PixelUnshuffle(2)

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        # self.planes = planes

    def forward(self, x):

        out = self.move11(x)
        out = self.prelu1(out)
        out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        # store_internal_output(out, f'internal_enc_conv_{out.shape[1]}.png')

        # shortcut = self.channel_pool(x)
        shortcut = x[:, :x.shape[1]//2, :, :]
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

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

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        # self.conv_ = conv3x3(inplanes, inplanes, stride)
        self.conv = conv3x3(inplanes, inplanes*2, stride)

        # self.conv1x1 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()
        # self.scale_ = LearnableScale(inplanes)

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        self.planes = planes


    def forward(self, x):
        out = self.move11(x)
        out = self.prelu1(out)
        out = self.move12(out)
        out = self.binary_activation(out)
        # out = self.conv_(out)

        # x = self.scale_(x)
        # x = x + out

        # out = self.move21(x)
        # out = self.prelu2(out)
        # out = self.move22(out)
        # out = self.binary_activation(out)
        out = self.conv(out)

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        shortcut = self.scale(x)
        # shortcut = shortcut.repeat(1, 2, 1, 1)
        shortcut = torch.cat((shortcut, shortcut), 1)
        # shortcut = self.conv1x1(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out

class BasicBlock_transpose_modified(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_transpose_modified, self).__init__()

        self.move11 = LearnableBias(inplanes//4)
        self.prelu1 = nn.PReLU(inplanes//4)
        self.move12 = LearnableBias(inplanes//4)
        self.binary_activation = BinaryActivation()

        self.conv = conv3x3(planes//2, planes, stride)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale(planes//2)

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        self.planes = planes


    def forward(self, x):
        x = self.px(x)

        out = self.move11(x)
        out = self.prelu1(out)
        out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)

        shortcut = self.scale(x)
        shortcut = torch.cat((shortcut, shortcut), 1)

        out = out + shortcut

        # out = self.px(out)

        # store_internal_output(out, f'internal_oup_{out.shape[1]}_dec.png')

        # print(out.shape)

        # assert out.shape[1] == self.planes

        return out



class BasicBlock_1(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_1, self).__init__()
        
        self.move11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = conv3x3(inplanes, planes, stride=2)
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
        # store_internal_output(out, f'internal_enc_conv_{out.shape[1]}.png')

        shortcut = x[:, :x.shape[1]//2, :, :]


        shortcut = self.px(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut
        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
        

class BasicBlock_transpose_1(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_transpose_1, self).__init__()

        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = conv3x3(inplanes, planes, stride)
        # self.conv1x1 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()

        self.move21 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move22 = LearnableBias(planes)

        self.planes = planes


    def forward(self, x):

        out = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.move11(out)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        out = self.move21(out)
        out = self.prelu2(out)
        out = self.move22(out)

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        shortcut = self.scale(x)
        shortcut = torch.cat((shortcut, shortcut), 1)

        shortcut = self.px(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut
        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out

class BasicBlock_transpose_2(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_transpose_2, self).__init__()

        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = conv3x3(inplanes, planes, stride)
        # self.conv2 = conv3x3(inplanes, planes, groups=inplanes//4)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()

        self.move21 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move22 = LearnableBias(planes)

        self.planes = planes


    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        out = self.move21(out)
        out = self.prelu2(out)
        out = self.move22(out)

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        # shortcut = torch.cat((shortcut, shortcut), 1)
        shortcut = x[:, :x.shape[1]//2, :, :]
        shortcut = self.scale(shortcut)

        # shortcut = self.px(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut
        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out


class BasicBlock_trivial(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_trivial, self).__init__()
        
        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = conv3x3(inplanes, inplanes//2, stride)
        # self.conv2 = conv1x1(inplanes//2, inplanes//2)
        # self.channel_pool = channel_wise_pool(inplanes, inplanes//2)
        self.px = nn.PixelUnshuffle(2)
        self.scale = LearnableScale()
        # self.scale2 = LearnableScale_per_channel(inplanes//2)

        self.move21 = LearnableBias(inplanes//2)
        self.prelu2 = nn.PReLU(inplanes//2)
        self.move22 = LearnableBias(inplanes//2)

        self.planes = planes

    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        out = self.move21(out)
        # out = self.scale2(out)
        out = self.prelu2(out)
        out = self.move22(out)
        # store_internal_output(out, f'internal_enc_conv_{out.shape[1]}.png')

        # shortcut = self.channel_pool(x)
        shortcut = x[:, :x.shape[1]//2, :, :]
        shortcut = self.scale(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
        

class BasicBlock_transpose_trivial(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_transpose_trivial, self).__init__()

        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        # self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.conv = conv3x3(inplanes, inplanes*2, stride)

        # self.conv2 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()
        # self.scale2 = LearnableScale_per_channel(inplanes*2)
        # self.scale_ = LearnableScale(inplanes)

        self.move21 = LearnableBias(inplanes*2)
        self.prelu2 = nn.PReLU(inplanes*2)
        # self.prelu3 = nn.PReLU(inplanes*2)
        self.move22 = LearnableBias(inplanes*2)

        self.planes = planes


    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        out = self.move21(out)
        # out = self.scale2(out)
        out = self.prelu2(out)
        out = self.move22(out)

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        shortcut = self.scale(x)
        # shortcut = shortcut.repeat(1, 2, 1, 1)
        shortcut = torch.cat((shortcut, shortcut), 1)
        # shortcut = self.conv1x1(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out

class BinaryBlock_transpose_1(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BinaryBlock_transpose_1, self).__init__()

        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = binaryconv3x3(inplanes, planes, stride)
        # self.conv1x1 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()

        self.move21 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move22 = LearnableBias(planes)

        self.planes = planes


    def forward(self, x):

        out = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.move11(out)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        out = self.move21(out)
        out = self.prelu2(out)
        out = self.move22(out)

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        shortcut = self.scale(x)
        shortcut = torch.cat((shortcut, shortcut), 1)

        shortcut = self.px(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut
        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out



class BinaryBlock_trivial(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BinaryBlock_trivial, self).__init__()
        
        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        self.conv = binaryconv3x3(inplanes, inplanes//2, stride)
        # self.conv2 = conv1x1(inplanes//2, inplanes//2)
        # self.channel_pool = channel_wise_pool(inplanes, inplanes//2)
        self.px = nn.PixelUnshuffle(2)
        self.scale = LearnableScale()
        # self.scale2 = LearnableScale_per_channel(inplanes//2)

        self.move21 = LearnableBias(inplanes//2)
        self.prelu2 = nn.PReLU(inplanes//2)
        self.move22 = LearnableBias(inplanes//2)

        self.planes = planes

    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        # out = self.scale2(out)
        out = self.move21(out)
        out = self.prelu2(out)
        out = self.move22(out)
        # store_internal_output(out, f'internal_enc_conv_{out.shape[1]}.png')

        # print(f'Mean_out: {torch.mean(out).item()}')
        # print(f'Std_out: {torch.std(out).item()}')

        # shortcut = self.channel_pool(x)
        shortcut = x[:, :x.shape[1]//2, :, :]
        shortcut = self.scale(shortcut)

        # print(f'Mean_shortcut: {torch.mean(shortcut).item()}')
        # print(f'Std_shortcut: {torch.std(shortcut).item()}')
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
        

class BinaryBlock_transpose_trivial(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BinaryBlock_transpose_trivial, self).__init__()

        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()

        # self.move21 = LearnableBias(inplanes)
        # self.prelu2 = nn.PReLU(inplanes)
        # self.move22 = LearnableBias(inplanes)

        # self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.conv = binaryconv3x3(inplanes, inplanes*2, stride)

        # self.conv2 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()
        # self.scale2 = LearnableScale_per_channel(inplanes*2)
        # self.scale_ = LearnableScale(inplanes)

        self.move21 = LearnableBias(inplanes*2)
        self.prelu2 = nn.PReLU(inplanes*2)
        # self.prelu3 = nn.PReLU(inplanes*2)
        self.move22 = LearnableBias(inplanes*2)

        self.planes = planes


    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv(out)
        # out = self.scale2(out)
        out = self.move21(out)
        out = self.prelu2(out)
        out = self.move22(out)

        # print(f'Mean_out: {torch.mean(out).item()}')
        # print(f'Std_out: {torch.std(out).item()}')
        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        shortcut = self.scale(x)
        # shortcut = shortcut.repeat(1, 2, 1, 1)
        shortcut = torch.cat((shortcut, shortcut), 1)

        # print(f'Mean_shortcut: {torch.mean(shortcut).item()}')
        # print(f'Std_shortcut: {torch.std(shortcut).item()}')
        # shortcut = self.conv1x1(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out

class RealBlock_thick(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(RealBlock_thick, self).__init__()
        
        self.prelu1 = nn.PReLU(inplanes)

        self.conv1 = conv3x3(inplanes, inplanes, bias=True)
        self.conv2 = conv3x3(inplanes, inplanes//2, stride, bias=True)
        self.px = nn.PixelUnshuffle(2)
        # self.scale1 = LearnableScale()
        # self.scale2 = LearnableScale()

        self.prelu4 = nn.PReLU(inplanes//2)

        self.planes = planes

        # self.se1 = AFlayer(inplanes)
        # self.se2 = AFlayer(inplanes*2)

    def forward(self, x, snr=0):

        out = self.conv1(x)
        out = self.prelu1(out)

        # x = self.scale1(x)
        out = out + x

        # out = self.se1(out, snr)

        x = out

        print(out.numel())

        out = self.conv2(out)
        out = self.prelu4(out)

        shortcut = x[:, :x.shape[1]//2, :, :]
        # shortcut = self.scale2(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        print(out.numel())

        # out = self.se2(out, snr)

        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')


        assert out.shape[1] == self.planes

        return out
        
class RealBlock_thick_transpose(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(RealBlock_thick_transpose, self).__init__()
        
        self.prelu1 = nn.PReLU(inplanes)

        self.conv1 = conv3x3(inplanes, inplanes, bias=True)
        self.conv2 = conv3x3(inplanes, inplanes*2, stride, bias=True)
        self.px = nn.PixelShuffle(2)
        # self.scale1 = LearnableScale()
        # self.scale2 = LearnableScale()

        self.prelu4 = nn.PReLU(inplanes*2)

        self.planes = planes

        # self.se1 = AFlayer(inplanes)
        # self.se2 = AFlayer(inplanes//2)

    def forward(self, x, snr):

        out = self.conv1(x)
        out = self.prelu1(out)

        # x = self.scale1(x)
        out = out + x

        # out = self.se1(out, snr)

        x = out
        print(out.numel())

        out = self.conv2(out)
        out = self.prelu4(out)


        shortcut = torch.cat((x, x), 1)
        # shortcut = self.scale2(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # out = self.se2(out, snr)

        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

        print(out.numel())

        assert out.shape[1] == self.planes

        return out
        



class BasicBlock_thick(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_thick, self).__init__()
        
        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.move21 = LearnableBias(inplanes)
        self.prelu2 = nn.PReLU(inplanes)
        self.move22 = LearnableBias(inplanes)

        self.conv1 = conv3x3(inplanes, inplanes)
        self.conv2 = conv3x3(inplanes, inplanes//2, stride)
        # self.channel_pool = channel_wise_pool(inplanes, inplanes//2)
        self.px = nn.PixelUnshuffle(2)
        self.scale1 = LearnableScale()
        self.scale2 = LearnableScale()

        self.move31 = LearnableBias(inplanes)
        self.move41 = LearnableBias(inplanes//2)
        self.prelu4 = nn.PReLU(inplanes//2)
        self.move42 = LearnableBias(inplanes//2)


        self.planes = planes

    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv1(out)
        out = self.move21(out)
        # out = self.scale2(out)
        out = self.prelu2(out)
        out = self.move22(out)
        # store_internal_output(out, f'internal_enc_conv_{out.shape[1]}.png')

        x = self.scale1(x)
        out = out + x
        x = out

        out = self.move31(x)
        out = self.binary_activation(out)
        out = self.conv2(out)
        out = self.move41(out)
        out = self.prelu4(out)
        out = self.move42(out)


        # shortcut = self.channel_pool(x)
        shortcut = x[:, :x.shape[1]//2, :, :]
        shortcut = self.scale2(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
        

class BasicBlock_transpose_thick(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_transpose_thick, self).__init__()

        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.move21 = LearnableBias(inplanes)
        self.prelu2 = nn.PReLU(inplanes)
        self.move22 = LearnableBias(inplanes)

        # self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.conv2 = conv3x3(inplanes, inplanes*2, stride)

        # self.conv2 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale1 = LearnableScale()
        self.scale2 = LearnableScale()
        # self.scale2 = LearnableScale_per_channel(inplanes*2)
        # self.scale_ = LearnableScale(inplanes)

        self.move31 = LearnableBias(inplanes)
        self.move41 = LearnableBias(inplanes*2)
        self.prelu4 = nn.PReLU(inplanes*2)
        self.move42 = LearnableBias(inplanes*2)


        self.planes = planes


    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv1(out)
        out = self.move21(out)
        # out = self.scale2(out)
        out = self.prelu2(out)
        out = self.move22(out)

        x = self.scale1(x)
        out = out + x
        x = out

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        out = self.move31(x)
        out = self.binary_activation(out)
        out = self.conv2(out)
        out = self.move41(out)
        out = self.prelu4(out)
        out = self.move42(out)

        shortcut = self.scale2(x)
        # shortcut = shortcut.repeat(1, 2, 1, 1)
        shortcut = torch.cat((shortcut, shortcut), 1)
        # shortcut = self.conv1x1(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out

class BinaryBlock_thick(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BinaryBlock_thick, self).__init__()
        
        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.move21 = LearnableBias(inplanes)
        self.prelu2 = nn.PReLU(inplanes)
        self.move22 = LearnableBias(inplanes)

        self.conv1 = binaryconv3x3(inplanes, inplanes)
        self.conv2 = binaryconv3x3(inplanes, inplanes//2, stride)
        # self.channel_pool = channel_wise_pool(inplanes, inplanes//2)
        self.px = nn.PixelUnshuffle(2)
        self.scale1 = LearnableScale()
        self.scale2 = LearnableScale()

        self.move31 = LearnableBias(inplanes)
        self.move41 = LearnableBias(inplanes//2)
        self.prelu4 = nn.PReLU(inplanes//2)
        self.move42 = LearnableBias(inplanes//2)


        self.planes = planes

    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv1(out)
        out = self.move21(out)
        # out = self.scale2(out)
        out = self.prelu2(out)
        out = self.move22(out)
        # store_internal_output(out, f'internal_enc_conv_{out.shape[1]}.png')

        x = self.scale1(x)
        out = out + x
        x = out

        out = self.move31(x)
        out = self.binary_activation(out)
        out = self.conv2(out)
        out = self.move41(out)
        out = self.prelu4(out)
        out = self.move42(out)


        # shortcut = self.channel_pool(x)
        shortcut = x[:, :x.shape[1]//2, :, :]
        shortcut = self.scale2(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
        

class BinaryBlock_transpose_thick(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BinaryBlock_transpose_thick, self).__init__()

        self.move11 = LearnableBias(inplanes)
        # self.prelu1 = nn.PReLU(inplanes)
        # self.move12 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.move21 = LearnableBias(inplanes)
        self.prelu2 = nn.PReLU(inplanes)
        self.move22 = LearnableBias(inplanes)

        # self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.conv1 = binaryconv3x3(inplanes, inplanes, stride)
        self.conv2 = binaryconv3x3(inplanes, inplanes*2, stride)

        # self.conv2 = conv1x1(inplanes*2, inplanes*2)
        self.px = nn.PixelShuffle(2)

        self.scale1 = LearnableScale()
        self.scale2 = LearnableScale()
        # self.scale2 = LearnableScale_per_channel(inplanes*2)
        # self.scale_ = LearnableScale(inplanes)

        self.move31 = LearnableBias(inplanes)
        self.move41 = LearnableBias(inplanes*2)
        self.prelu4 = nn.PReLU(inplanes*2)
        self.move42 = LearnableBias(inplanes*2)


        self.planes = planes


    def forward(self, x):

        out = self.move11(x)
        # out = self.prelu1(out)
        # out = self.move12(out)
        out = self.binary_activation(out)
        out = self.conv1(out)
        out = self.move21(out)
        # out = self.scale2(out)
        out = self.prelu2(out)
        out = self.move22(out)

        x = self.scale1(x)
        out = out + x
        x = out

        # store_internal_output(out, f'internal_dec_conv_{out.shape[1]}.png')

        out = self.move31(x)
        out = self.binary_activation(out)
        out = self.conv2(out)
        out = self.move41(out)
        out = self.prelu4(out)
        out = self.move42(out)

        shortcut = self.scale2(x)
        # shortcut = shortcut.repeat(1, 2, 1, 1)
        shortcut = torch.cat((shortcut, shortcut), 1)
        # shortcut = self.conv1x1(shortcut)
        # store_internal_output(shortcut, f'internal_dec_shortcut_{shortcut.shape[1]}.png')

        out = out + shortcut

        out = self.px(out)

        # store_internal_output(out, f'internal_dec_oup_{out.shape[1]}.png')

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


class AdaBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(AdaBlock, self).__init__()

        # self.binary_activation = BA_ada()


        self.conv = conv_ada(inplanes, inplanes//2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1)
        self.move1 = LearnableBias(inplanes//2)
        # self.maxout = Maxout(inplanes//2)
        self.prelu = nn.PReLU(inplanes//2)
        self.move2 = LearnableBias(inplanes//2)

        self.px = nn.PixelUnshuffle(2)


        self.scale = LearnableScale()

        self.planes = planes

    def forward(self, x):
        
        out = x
        # out = self.binary_activation(x)
        out = self.conv(out)
        out = self.move1(out)
        out = self.prelu(out)
        # out = self.maxout(out)
        out = self.move2(out)

        shortcut = x[:, :x.shape[1]//2, :, :]
        shortcut = self.scale(shortcut)

        out = out + shortcut

        out = self.px(out)

        assert out.shape[1] == self.planes

        return out
    
class AdaBlock_transpose(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(AdaBlock_transpose, self).__init__()

        # self.binary_activation = BA_ada()

        self.conv = conv_ada(inplanes, inplanes*2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1)
        self.move1 = LearnableBias(inplanes*2)
        # self.maxout = Maxout(inplanes*2)
        self.prelu = nn.PReLU(inplanes*2)
        self.move2 = LearnableBias(inplanes*2)

        self.px = nn.PixelShuffle(2)

        self.scale = LearnableScale()

        self.planes = planes

    def forward(self, x):
        
        out = x
        # out = self.binary_activation(x)
        out = self.conv(out)
        out = self.move1(out)
        out = self.prelu(out)
        # out = self.maxout(out)
        out = self.move2(out)

        shortcut = torch.cat((x, x), 1)
        shortcut = self.scale(shortcut)

        out = out + shortcut

        out = self.px(out)

        assert out.shape[1] == self.planes

        return out

class AdaBlock_thick(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(AdaBlock_thick, self).__init__()

        self.conv1 = conv_ada(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1)
        # self.maxout1 = Maxout(inplanes)
        self.move1 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move2 = LearnableBias(inplanes)


        self.conv2 = conv_ada(inplanes, inplanes//2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1)
        # self.maxout2 = Maxout(inplanes//2)
        self.move3 = LearnableBias(inplanes//2)
        self.prelu2 = nn.PReLU(inplanes//2)
        self.move4 = LearnableBias(inplanes//2)

        self.px = nn.PixelUnshuffle(2)


        # self.scale1 = LearnableScale()
        # self.scale2 = LearnableScale()

        self.planes = planes

    def forward(self, x):
        
        out = x
        out = self.conv1(out)
        # out = self.maxout1(out)
        out = self.move1(out)
        out = self.prelu1(out)
        out = self.move2(out)

        # shortcut = self.scale1(x)
        out = out + x
        x = out

        out = self.conv2(out)
        # out = self.maxout2(out)
        out = self.move3(out)
        out = self.prelu2(out)
        out = self.move4(out)

        shortcut = x[:, :x.shape[1]//2, :, :]
        # shortcut = self.scale2(shortcut)

        out = out + shortcut

        out = self.px(out)

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
    
class AdaBlock_thick_transpose(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(AdaBlock_thick_transpose, self).__init__()

        self.conv1 = conv_ada(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1)
        # self.maxout1 = Maxout(inplanes)
        self.move1 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move2 = LearnableBias(inplanes)


        self.conv2 = conv_ada(inplanes, inplanes*2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1)
        # self.maxout2 = Maxout(inplanes*2)
        self.move3 = LearnableBias(inplanes*2)
        self.prelu2 = nn.PReLU(inplanes*2)
        self.move4 = LearnableBias(inplanes*2)

        self.px = nn.PixelShuffle(2)


        # self.scale1 = LearnableScale()
        # self.scale2 = LearnableScale()

        self.planes = planes

    def forward(self, x):
        
        out = x
        # out = self.binary_activation(x)
        out = self.conv1(out)
        # out = self.maxout1(out)
        out = self.move1(out)
        out = self.prelu1(out)
        out = self.move2(out)

        # shortcut = self.scale1(x)
        out = out + x
        x = out

        out = self.conv2(out)
        # out = self.maxout2(out)
        out = self.move3(out)
        out = self.prelu2(out)
        out = self.move4(out)

        shortcut = torch.cat((x, x), 1)
        # shortcut = self.scale2(shortcut)

        out = out + shortcut

        out = self.px(out)

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
    
class BDJSCC(nn.Module):
    def __init__(self, init_channels=64, channel_type='awgn', snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None):
        super(BDJSCC, self).__init__()
        self.firstconv = firstconv(3, init_channels)
        self.layer1 = RealBlock_thick(init_channels, init_channels*2)
        self.layer2 = RealBlock_thick(init_channels*2, init_channels*4)
        self.layer3 = RealBlock_thick(init_channels*4, init_channels*8)
        self.layer3_ = RealBlock_thick(init_channels*8, init_channels*16)
        self.channel = Channel(channel_type=channel_type)
        self.layer4_ = RealBlock_thick_transpose(init_channels*16, init_channels*8)
        self.layer4 = RealBlock_thick_transpose(init_channels*8, init_channels*4)
        self.layer5 = RealBlock_thick_transpose(init_channels*4, init_channels*2)
        self.layer6 = RealBlock_thick_transpose(init_channels*2, init_channels)
        self.lastconv = lastconv(init_channels, 3)
        self.final_activation = nn.Sigmoid()

        self.snr_db = snr_db
        self.h_real = h_real
        self.h_imag = h_imag
        self.b_prob = b_prob
        self.b_stddev = b_stddev


    def forward(self, x):
        flops_com = 0
        if self.snr_db is None:
            snr = torch.randint(1, 20, (1,)).item()
        else:
            snr = self.snr_db
        out = self.firstconv(x)
        flops_com += out.numel()*3
        out = self.layer1(out, snr)
        flops_com += out.numel()*3
        out = self.layer2(out, snr)
        flops_com += out.numel()*3
        out = self.layer3(out, snr)
        flops_com += out.numel()*3
        out = self.layer3_(out, snr)
        out = self.channel(out, snr, self.h_real, self.h_imag, self.b_prob, self.b_stddev)
        flops_com += out.numel()*6
        out = self.layer4_(out, snr)
        flops_com += out.numel()*6
        out = self.layer4(out, snr)
        flops_com += out.numel()*6
        out = self.layer5(out, snr)
        flops_com += out.numel()*6
        out = self.layer6(out, snr)
        out = self.lastconv(out)
        out = self.final_activation(out)
        print(f'FLOPS: {flops_com}')
        flops_com += out.numel()*4
        print(f'FLOPS: {flops_com}')

        return out
    

class BDJSCC_Binary(nn.Module):
    def __init__(self, init_channels=32):
        super(BDJSCC_Binary, self).__init__()
        self.firstconv = firstconv(3, init_channels)
        self.layer1 = BinaryBlock_thick(init_channels, init_channels*2)
        self.layer2 = BinaryBlock_thick(init_channels*2, init_channels*4)
        self.layer3 = BinaryBlock_thick(init_channels*4, init_channels*8)
        # self.layer3_ = BinaryBlock_trivial(init_channels*8, init_channels*16)
        # self.layer4_ = BinaryBlock_transpose_trivial(init_channels*16, init_channels*8)
        self.layer4 = BinaryBlock_transpose_thick(init_channels*8, init_channels*4)
        self.layer5 = BinaryBlock_transpose_thick(init_channels*4, init_channels*2)
        self.layer6 = BinaryBlock_transpose_thick(init_channels*2, init_channels)
        self.lastconv = lastconv(init_channels, 3)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        out = self.firstconv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer3_(out)

        # out = self.layer4_(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.lastconv(out)
        out = self.final_activation(out)

        return out


class BDJSCC_ada(nn.Module):
    def __init__(self, init_channels=64, channel_type='awgn', snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None):
        super(BDJSCC_ada, self).__init__()
        self.firstconv = firstconv(3, init_channels)
        self.layer1 = AdaBlock_thick(init_channels, init_channels*2)
        self.layer2 = AdaBlock_thick(init_channels*2, init_channels*4)
        self.layer3 = AdaBlock_thick(init_channels*4, init_channels*8)
        self.layer3_ = AdaBlock_thick(init_channels*8, init_channels*16)
        self.channel = Channel(channel_type=channel_type)
        self.layer4_ = AdaBlock_thick_transpose(init_channels*16, init_channels*8)
        self.layer4 = AdaBlock_thick_transpose(init_channels*8, init_channels*4)
        self.layer5 = AdaBlock_thick_transpose(init_channels*4, init_channels*2)
        self.layer6 = AdaBlock_thick_transpose(init_channels*2, init_channels)
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