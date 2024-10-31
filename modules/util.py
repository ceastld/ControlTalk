from torch import nn

import torch.nn.functional as F
import torch

from modules.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from modules.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

import torch.nn.utils.spectral_norm as spectral_norm
import re


def kp2gaussian(kp, spatial_size, kp_variance, coordinate_grid):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid_2d(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def make_coordinate_grid(spatial_size, type, device='cuda'):
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
   
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)
    meshed = meshed.to(device)

    return meshed


class ResBottleneck(nn.Module):
    def __init__(self, in_features, stride):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features, kernel_size=1)
        self.norm1 = BatchNorm2d(in_features//4, affine=True)
        self.norm2 = BatchNorm2d(in_features//4, affine=True)
        self.norm3 = BatchNorm2d(in_features, affine=True)

        self.stride = stride
        if self.stride != 1:
            self.skip = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=stride)
            self.norm4 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.stride != 1:
            x = self.skip(x)
            x = self.norm4(x)
        out += x
        out = F.relu(out)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=(1, 2, 2))
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
           
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

class KPHourglass(nn.Module):
    """
    Hourglass architecture.
    """ 

    def __init__(self, block_expansion, in_features, reshape_features, reshape_depth, num_blocks=3, max_features=256):
        super(KPHourglass, self).__init__()
        
        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.down_blocks.add_module('down'+ str(i), DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                                                   min(max_features, block_expansion * (2 ** (i + 1))),
                                                                   kernel_size=3, padding=1))

        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=reshape_features, kernel_size=1)

        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_filters = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module('up'+ str(i), UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def forward(self, x):
        out = self.down_blocks(x)
        out = self.conv(out)
        bs, c, h, w = out.shape
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        out = self.up_blocks(out)

        return out
        


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1

        meshgrids = torch.meshgrid(
        [
            torch.arange(size, device="cuda", dtype=torch.float32) # float32
            for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


if __name__ == "__main__":
    from torchinfo import summary
    block_expansion = 32
    max_features = 1024
    num_blocks = 5
    num_kp = 15
    compress = 4
    # model = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)
    # summary(model, input_size=(1, (num_kp+1)*(compress+1), 16, 128, 128), device='cpu')

    model = KPHourglass(block_expansion=32, in_features=3, reshape_features=16384, reshape_depth=16, max_features=1024, num_blocks=5)
    summary(model, input_size=(1, 3, 512, 512), device='cpu')

# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# Hourglass                                          [1, 112, 16, 128, 128]    --
# ├─Encoder: 1-1                                     [1, 80, 16, 128, 128]     --
# │    └─ModuleList: 2-1                             --                        --
# │    │    └─DownBlock3d: 3-1                       [1, 64, 16, 64, 64]       138,432
# │    │    └─DownBlock3d: 3-2                       [1, 128, 16, 32, 32]      221,568
# │    │    └─DownBlock3d: 3-3                       [1, 256, 16, 16, 16]      885,504
# │    │    └─DownBlock3d: 3-4                       [1, 512, 16, 8, 8]        3,540,480
# │    │    └─DownBlock3d: 3-5                       [1, 1024, 16, 4, 4]       14,158,848
# ├─Decoder: 1-2                                     [1, 112, 16, 128, 128]    --
# │    └─ModuleList: 2-2                             --                        --
# │    │    └─UpBlock3d: 3-6                         [1, 512, 16, 8, 8]        14,157,312
# │    │    └─UpBlock3d: 3-7                         [1, 256, 16, 16, 16]      7,078,656
# │    │    └─UpBlock3d: 3-8                         [1, 128, 16, 32, 32]      1,769,856
# │    │    └─UpBlock3d: 3-9                         [1, 64, 16, 64, 64]       442,560
# │    │    └─UpBlock3d: 3-10                        [1, 32, 16, 128, 128]     110,688
# │    └─Conv3d: 2-3                                 [1, 112, 16, 128, 128]    338,800
# │    └─SynchronizedBatchNorm3d: 2-4                [1, 112, 16, 128, 128]    224
# ====================================================================================================
# Total params: 42,842,928
# Trainable params: 42,842,928
# Non-trainable params: 0
# Total mult-adds (G): 313.54
# ====================================================================================================
# Input size (MB): 83.89
# Forward/backward pass size (MB): 1249.90
# Params size (MB): 171.37
# Estimated Total Size (MB): 1505.16
# ====================================================================================================

# KPHourglass
# ===============================================================================================
# Layer (type:depth-idx)                        Output Shape              Param #
# ===============================================================================================
# KPHourglass                                   [1, 32, 16, 512, 512]     --
# ├─Sequential: 1-1                             [1, 1024, 16, 16]         --
# │    └─DownBlock2d: 2-1                       [1, 64, 256, 256]         --
# │    │    └─Conv2d: 3-1                       [1, 64, 512, 512]         1,792
# │    │    └─SynchronizedBatchNorm2d: 3-2      [1, 64, 512, 512]         128
# │    │    └─AvgPool2d: 3-3                    [1, 64, 256, 256]         --
# │    └─DownBlock2d: 2-2                       [1, 128, 128, 128]        --
# │    │    └─Conv2d: 3-4                       [1, 128, 256, 256]        73,856
# │    │    └─SynchronizedBatchNorm2d: 3-5      [1, 128, 256, 256]        256
# │    │    └─AvgPool2d: 3-6                    [1, 128, 128, 128]        --
# │    └─DownBlock2d: 2-3                       [1, 256, 64, 64]          --
# │    │    └─Conv2d: 3-7                       [1, 256, 128, 128]        295,168
# │    │    └─SynchronizedBatchNorm2d: 3-8      [1, 256, 128, 128]        512
# │    │    └─AvgPool2d: 3-9                    [1, 256, 64, 64]          --
# │    └─DownBlock2d: 2-4                       [1, 512, 32, 32]          --
# │    │    └─Conv2d: 3-10                      [1, 512, 64, 64]          1,180,160
# │    │    └─SynchronizedBatchNorm2d: 3-11     [1, 512, 64, 64]          1,024
# │    │    └─AvgPool2d: 3-12                   [1, 512, 32, 32]          --
# │    └─DownBlock2d: 2-5                       [1, 1024, 16, 16]         --
# │    │    └─Conv2d: 3-13                      [1, 1024, 32, 32]         4,719,616
# │    │    └─SynchronizedBatchNorm2d: 3-14     [1, 1024, 32, 32]         2,048
# │    │    └─AvgPool2d: 3-15                   [1, 1024, 16, 16]         --
# ├─Conv2d: 1-2                                 [1, 16384, 16, 16]        16,793,600
# ├─Sequential: 1-3                             [1, 32, 16, 512, 512]     --
# │    └─UpBlock3d: 2-6                         [1, 512, 16, 32, 32]      --
# │    │    └─Conv3d: 3-16                      [1, 512, 16, 32, 32]      14,156,288
# │    │    └─SynchronizedBatchNorm3d: 3-17     [1, 512, 16, 32, 32]      1,024
# │    └─UpBlock3d: 2-7                         [1, 256, 16, 64, 64]      --
# │    │    └─Conv3d: 3-18                      [1, 256, 16, 64, 64]      3,539,200
# │    │    └─SynchronizedBatchNorm3d: 3-19     [1, 256, 16, 64, 64]      512
# │    └─UpBlock3d: 2-8                         [1, 128, 16, 128, 128]    --
# │    │    └─Conv3d: 3-20                      [1, 128, 16, 128, 128]    884,864
# │    │    └─SynchronizedBatchNorm3d: 3-21     [1, 128, 16, 128, 128]    256
# │    └─UpBlock3d: 2-9                         [1, 64, 16, 256, 256]     --
# │    │    └─Conv3d: 3-22                      [1, 64, 16, 256, 256]     221,248
# │    │    └─SynchronizedBatchNorm3d: 3-23     [1, 64, 16, 256, 256]     128
# │    └─UpBlock3d: 2-10                        [1, 32, 16, 512, 512]     --
# │    │    └─Conv3d: 3-24                      [1, 32, 16, 512, 512]     55,328
# │    │    └─SynchronizedBatchNorm3d: 3-25     [1, 32, 16, 512, 512]     64
# ===============================================================================================
# Total params: 41,927,072
# Trainable params: 41,927,072
# Non-trainable params: 0
# Total mult-adds (T): 1.18
# ===============================================================================================
# Input size (MB): 3.15
# Forward/backward pass size (MB): 4714.40
# Params size (MB): 167.71
# Estimated Total Size (MB): 4885.25
# ===============================================================================================