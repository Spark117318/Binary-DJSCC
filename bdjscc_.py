import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import PIL 
import os
from adalib import BinaryActivation as BA_ada
from adalib import AdaBin_Conv2d as conv_ada
from channel import Channel
from SEblock import AFlayer
from ABNet import Q_PReLU

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



class firstconv(nn.Module):
    def __init__(self, inp, oup, size=3, binary=False):
        super(firstconv, self).__init__()

        if binary:
            self.conv1 = conv_ada(inp, oup//4, kernel_size=size, stride=1, padding=size//2, bias=False, w_bit=1, a_bit=0)
        else:
            self.conv1 = nn.Conv2d(inp, oup//4, size, 1, size//2, bias=True)
        self.px1 = nn.PixelUnshuffle(2)
        self.prelu = nn.PReLU(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.px1(out)
        out = self.prelu(out)

        return out


class firstconv_2(nn.Module):
    def __init__(self, inplanes=3, planes=16, size=3, stride=1, ka=1, alpha=0.25, kw=0.2, binary=0, QReLU=False):
        super(firstconv_2, self).__init__()

        # self.move1_ = LearnableBias(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=size, stride=stride, padding=size//2, bias=True, groups=inplanes)
        # self.move1 = LearnableBias(planes)

        if QReLU:
            self.prelu1 = Q_PReLU(inplanes)
        else:
            self.prelu1 = nn.PReLU(inplanes)

        # self.move2 = LearnableBias(planes)

        self.conv2 = nn.Conv2d(inplanes+1, planes, kernel_size=1, stride=stride, padding=1//2, bias=True)
        
        if QReLU:
            self.prelu2 = Q_PReLU(planes)
        else:
            self.prelu2 = nn.PReLU(planes)


        self.px = nn.PixelUnshuffle(2)


        self.planes = planes

    def forward(self, x, snr):
        
        # out = self.move1_(x)
        out = self.conv1(x)
        # out = self.move1(out)
        out = self.prelu1(out)
        # out = self.move2(out)

        # Reshape snr from (batch_size,) to (batch_size, 1, height, width)
        _, _, height, width = x.shape
        snr_layer = snr.view(-1, 1, 1, 1).expand(-1, 1, height, width)
        snr_layer = snr_layer.to(x.device)
        out = torch.cat((snr_layer, out), 1)

        out = self.conv2(out)
        out = self.prelu2(out)

        assert out.shape[1] == self.planes

        out = self.px(out)


        return out

class firstdup(nn.Module):
    def __init__(self, inp, oup):
        super(firstdup, self).__init__()
        oup = oup // 4
        self.inp = inp
        self.oup = oup
        # Calculate repeat factor and remainder
        self.repeat_factor = oup // inp
        self.remainder = oup % inp

        self.px = nn.PixelUnshuffle(2)

    def forward(self, x):
        # Input shape: [batch_size, inp, height, width]
        
        # Handle the main repeats
        if self.repeat_factor > 0:
            out = x.repeat(1, self.repeat_factor, 1, 1)
        else:
            out = torch.tensor([], device=x.device)
            
        # Handle the remainder (if any)
        if self.remainder > 0:
            remainder_slice = x[:, :self.remainder, :, :]
            # Concatenate the repeated tensor with the remainder
            if self.repeat_factor > 0:
                out = torch.cat([out, remainder_slice], dim=1)
            else:
                out = remainder_slice

        out = self.px(out)
        
        return out



class lastconv(nn.Module):
    def __init__(self, inp, oup, size=3, binary=False):
        super(lastconv, self).__init__()

        if binary:
            self.conv1 = conv_ada(inp, oup*4, kernel_size=size, stride=1, padding=size//2, bias=False, w_bit=1, a_bit=0)
        else:
            self.conv1 = nn.Conv2d(inp, oup*4, size, 1, size//2, bias=False)
        self.px1 = nn.PixelShuffle(2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.px1(out)

        return out


class lastconv_2(nn.Module):
    def __init__(self, inp, oup, size=3, binary=False):
        super(lastconv_2, self).__init__()

        self.px1 = nn.PixelShuffle(2)
        if binary:
            self.conv1 = conv_ada(inp//4, oup, kernel_size=size, stride=1, padding=size//2, bias=False, w_bit=1, a_bit=0)
        else:
            self.conv1 = nn.Conv2d(inp//4, oup, size, 1, size//2, bias=False)

    def forward(self, x):

        out = self.px1(x)
        out = self.conv1(out)

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
        # self.se2_ = AFlayer(inplanes//2)

    def forward(self, x, snr=0):

        out = self.conv1(x)
        out = self.prelu1(out)

        # x = self.scale1(x)
        # out = self.se1(out, snr)
        out = out + x


        x = out

        # print(out.numel())

        out = self.conv2(out)
        shortcut = x[:, :x.shape[1]//2, :, :]
        # shortcut = self.scale2(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')
        # out = self.se2_(out, snr)


        out = out + shortcut
        out = self.prelu4(out)


        out = self.px(out)

        # print(out.numel())


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
        # self.se2_ = AFlayer(inplanes*2)

    def forward(self, x, snr):

        out = self.conv1(x)
        out = self.prelu1(out)

        # x = self.scale1(x)
        # out = self.se1(out, snr)
        out = out + x


        x = out
        # print(out.numel())

        out = self.conv2(out)
        shortcut = torch.cat((x, x), 1)
        # shortcut = self.scale2(shortcut)
        # store_internal_output(shortcut, f'internal_enc_shortcut_{shortcut.shape[1]}.png')

        # out = self.se2_(out, snr)
        out = out + shortcut
        out = self.prelu4(out)



        out = self.px(out)


        # store_internal_output(out, f'internal_enc_oup_{out.shape[1]}.png')

        # print(out.numel())

        assert out.shape[1] == self.planes

        return out
        



class AdaBlock_thick(nn.Module):
    def __init__(self, inplanes, planes, stride=1, ka=1, alpha=0.25, kw=0.2, binary=1, QReLU=True):
        super(AdaBlock_thick, self).__init__()

        self.move1_ = LearnableBias(inplanes)
        self.conv1 = conv_ada(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=binary, a_bit=1, ka=ka, kw=kw)
        # self.conv1_ = conv_ada(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1, a_bit=1)
        # self.maxout1 = Maxout(inplanes)
        self.move1 = LearnableBias(inplanes)

        if QReLU:
            self.prelu1 = Q_PReLU(inplanes)
        else:
            self.prelu1 = nn.PReLU(inplanes)

        self.move2 = LearnableBias(inplanes)

        self.move2_ = LearnableBias(inplanes)
        self.conv2 = conv_ada(inplanes, inplanes//2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=binary, a_bit=1, ka=ka, kw=kw)
        # self.conv2_ = conv_ada(inplanes, inplanes//2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1, a_bit=1)
        # self.maxout2 = Maxout(inplanes//2)
        self.move3 = LearnableBias(inplanes//2)

        if QReLU:
            self.prelu2 = Q_PReLU(inplanes//2)
        else:
            self.prelu2 = nn.PReLU(inplanes//2)

        self.move4 = LearnableBias(inplanes//2)

        self.px = nn.PixelUnshuffle(2)


        # self.scale1 = LearnableScale()
        # self.scale2 = LearnableScale()

        self.planes = planes

    def forward(self, x):
        
        # out1 = self.conv1(x)
        # out2 = self.conv1_(x)
        # out = out1 + out2
        out = self.move1_(x)
        out = self.conv1(out)
        x = out + x
        # out = self.maxout1(out)
        out = self.move1(out)
        out = self.prelu1(out)
        out = self.move2(out)

        # shortcut = self.scale1(x)
        # print(f"input_std: {torch.std(x).item()}")
        # print(f"output_std: {torch.std(out).item()}")

        # out1 = self.conv2(x)
        # out2 = self.conv2_(x)
        # out = out1 + out2
        out = self.move2_(x)
        out = self.conv2(out)
        shortcut = x[:, :x.shape[1]//2, :, :]
        # shortcut = self.scale2(shortcut)

        # print(f"input_std: {torch.std(shortcut).item()}")
        # print(f"output_std: {torch.std(out).item()}")
        out = out + shortcut
        # out = self.maxout2(out)
        out = self.move3(out)
        out = self.prelu2(out)
        out = self.move4(out)


        out = self.px(out)

        # print(out.shape)

        assert out.shape[1] == self.planes

        return out
    
class AdaBlock_thick_transpose(nn.Module):
    def __init__(self, inplanes, planes, stride=1, ka=1, alpha=0.25, kw=0.2, binary=1, QReLU=True):
        super(AdaBlock_thick_transpose, self).__init__()

        self.move1_ = LearnableBias(inplanes)
        self.conv1 = conv_ada(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=binary, a_bit=1, ka=ka, kw=kw)
        # self.conv1_ = conv_ada(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1, a_bit=1)
        # self.maxout1 = Maxout(inplanes)
        self.move1 = LearnableBias(inplanes)

        if QReLU:
            self.prelu1 = Q_PReLU(inplanes)
        else:
            self.prelu1 = nn.PReLU(inplanes)

        self.move2 = LearnableBias(inplanes)

        self.move2_ = LearnableBias(inplanes)
        self.conv2 = conv_ada(inplanes, inplanes*2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=binary, a_bit=1, ka=ka+alpha, kw=kw)
        # self.conv2_ = conv_ada(inplanes, inplanes*2, kernel_size=3, stride=stride, padding=1, bias=False, w_bit=1, a_bit=1)
        # self.maxout2 = Maxout(inplanes*2)
        self.move3 = LearnableBias(inplanes*2)

        if QReLU:
            self.prelu2 = Q_PReLU(inplanes*2)
        else:
            self.prelu2 = nn.PReLU(inplanes*2)
        self.move4 = LearnableBias(inplanes*2)

        self.px = nn.PixelShuffle(2)


        # self.scale1 = LearnableScale()
        # self.scale2 = LearnableScale()

        self.planes = planes

    def forward(self, x):
        
        # out1 = self.conv1(x)
        # out2 = self.conv1_(x)
        # out = out1 + out2
        out = self.move1_(x)
        out = self.conv1(out)
        x = out + x
        # out = self.maxout1(out)
        out = self.move1(out)
        out = self.prelu1(out)
        out = self.move2(out)

        # shortcut = self.scale1(x)
        # print(f"input_std: {torch.std(x).item()}")
        # print(f"output_std: {torch.std(out).item()}")

        # out1 = self.conv2(x)
        # out2 = self.conv2_(x)
        # out = out1 + out2
        out = self.move2_(x)
        out = self.conv2(out)
        shortcut = torch.cat((x, x), 1)
        # shortcut = self.scale2(shortcut)

        # print(f"input_std: {torch.std(shortcut).item()}")
        # print(f"output_std: {torch.std(out).item()}")
        out = out + shortcut
        # out = self.maxout2(out)
        out = self.move3(out)
        out = self.prelu2(out)
        out = self.move4(out)



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
        # flops_com = 0
        snr = self.channel.get_snr(x, self.snr_db)
        out = self.firstconv(x)
        # flops_com += out.numel()*3
        out = self.layer1(out, snr)
        # flops_com += out.numel()*3
        out = self.layer2(out, snr)
        # flops_com += out.numel()*3
        out = self.layer3(out, snr)
        # flops_com += out.numel()*3
        out = self.layer3_(out, snr)
        out = self.channel(out, snr, self.h_real, self.h_imag, self.b_prob, self.b_stddev)
        # flops_com += out.numel()*6
        out = self.layer4_(out, snr)
        # flops_com += out.numel()*6
        out = self.layer4(out, snr)
        # flops_com += out.numel()*6
        out = self.layer5(out, snr)
        # flops_com += out.numel()*6
        out = self.layer6(out, snr)
        out = self.lastconv(out)
        out = self.final_activation(out)
        # print(f'FLOPS: {flops_com}')
        # flops_com += out.numel()*4
        # print(f'FLOPS: {flops_com}')

        return out
    



class BDJSCC_ada(nn.Module):
    def __init__(self, init_channels=64, channel_type='awgn', snr_db=None, h_real=None, h_imag=None, b_prob=None, b_stddev=None, expected_var=1, alpha=0.25, enable_firstdup=False, kernel_size=3):
        super(BDJSCC_ada, self).__init__()
        if enable_firstdup:
            self.firstconv = firstdup(3, init_channels)
        else:
            self.firstconv = firstconv(3, init_channels, size=kernel_size)
        ka = expected_var*2
        self.layer1 = AdaBlock_thick(init_channels, init_channels*2, ka=ka, alpha=alpha)
        # ka += 2*alpha
        self.layer2 = AdaBlock_thick(init_channels*2, init_channels*4, ka=ka, alpha=alpha)
        # ka += 2*alpha
        self.layer3 = AdaBlock_thick(init_channels*4, init_channels*8, ka=ka, alpha=alpha)
        # ka += 2*alpha
        self.layer3_ = AdaBlock_thick(init_channels*8, init_channels*16, ka=ka, alpha=alpha)
        self.channel = Channel(channel_type=channel_type)
        self.scale = LearnableScale()
        ka = expected_var/4
        self.layer4_ = AdaBlock_thick_transpose(init_channels*16, init_channels*8, ka=ka, alpha=alpha)
        # ka += 2*alpha
        self.layer4 = AdaBlock_thick_transpose(init_channels*8, init_channels*4, ka=ka, alpha=alpha)
        # ka += 2*alpha
        self.layer5 = AdaBlock_thick_transpose(init_channels*4, init_channels*2, ka=ka, alpha=alpha)
        # ka += 2*alpha
        self.layer6 = AdaBlock_thick_transpose(init_channels*2, init_channels)
        self.lastconv = lastconv(init_channels, 3, size=kernel_size, binary=0)
        self.final_activation = nn.Sigmoid()
        # self.final_activation = nn.Hardtanh(0, 1)

        self.snr_db = snr_db
        self.h_real = h_real
        self.h_imag = h_imag
        self.b_prob = b_prob
        self.b_stddev = b_stddev

    def forward(self, x):
        snr = self.channel.get_snr(x, self.snr_db)
        out = self.firstconv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer3_(out)
        out = self.channel(out, self.snr_db, self.h_real, self.h_imag, self.b_prob, self.b_stddev)
        out = self.scale(out)
        out = self.layer4_(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.lastconv(out)
        out = self.final_activation(out)

        return out