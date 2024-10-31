import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
from modules.dense_motion import DenseMotionNetwork

class OcclusionAwareGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False, decoder_params=None):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.resblocks_2d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_2d.add_module('2dr' + str(i), ResBlock2d(out_features, kernel_size=3, padding=1))

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, image_channel, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print("out 1: ", out.shape)              # [1, 512, 128, 128])
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)
        # print("feature_3d: ", feature_3d.shape)  # ([1, 32, 16, 128, 128])

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            # output_dict["deformed"] = self.deform_input(source_image, deformation)  # 3d deformation cannot deform 2d image

        # Decoding part
        out = self.resblocks_2d(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = torch.sigmoid(out)
        output_dict["prediction"] = out

        return output_dict


class SPADEDecoder(nn.Module):
    def __init__(self, G_mid_full=None):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.fc         = nn.Conv2d(ic, 2 * ic, 3, padding=1)                 # 256 -> 512
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)  # 512 512 256
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # 
        self.G_mid_full = G_mid_full
        if self.G_mid_full:
            self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
            self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
            self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
            self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)

        self.up_0       = SPADEResnetBlock(2 * ic, ic,     norm_G, label_nc)  # 512 256 256
        self.up_1       = SPADEResnetBlock(ic,     oc,     norm_G, label_nc)  # 256 64  256
        self.conv_img   = nn.Conv2d(oc, 3, 3, padding=1)
        self.up         = nn.Upsample(scale_factor=2)
        
    def forward(self, feature):
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        # 
        if self.G_mid_full:
            x = self.G_middle_2(x, seg)
            x = self.G_middle_3(x, seg)
            x = self.G_middle_4(x, seg)
            x = self.G_middle_5(x, seg)
        # 
        x = self.up(x)                
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)                
        x = self.up_1(x, seg)         # 64, 256, 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        x = torch.sigmoid(x)
        
        return x

class OcclusionAwareSPADEGenerator(nn.Module):

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False, decoder_params=None):
        super(OcclusionAwareSPADEGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.decoder = SPADEDecoder(**decoder_params)

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, source_image, kp_driving, kp_source):
      
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        # Decoding part
        out = self.decoder(out)

        output_dict["prediction"] = out

        return output_dict

if __name__ == "__main__":
    pass

# OcclusionAwareGenerator
# ==============================================================================================================
# Layer (type:depth-idx)                                       Output Shape              Param #
# ==============================================================================================================
# OcclusionAwareGenerator                                      [1, 3, 512, 512]          --
# ├─SameBlock2d: 1-1                                           [1, 64, 512, 512]         --
# │    └─Conv2d: 2-1                                           [1, 64, 512, 512]         9,472
# │    └─SynchronizedBatchNorm2d: 2-2                          [1, 64, 512, 512]         128
# │    └─ReLU: 2-3                                             [1, 64, 512, 512]         --
# ├─ModuleList: 1-2                                            --                        --
# │    └─DownBlock2d: 2-4                                      [1, 128, 256, 256]        --
# │    │    └─Conv2d: 3-1                                      [1, 128, 512, 512]        73,856
# │    │    └─SynchronizedBatchNorm2d: 3-2                     [1, 128, 512, 512]        256
# │    │    └─AvgPool2d: 3-3                                   [1, 128, 256, 256]        --
# │    └─DownBlock2d: 2-5                                      [1, 256, 128, 128]        --
# │    │    └─Conv2d: 3-4                                      [1, 256, 256, 256]        295,168
# │    │    └─SynchronizedBatchNorm2d: 3-5                     [1, 256, 256, 256]        512
# │    │    └─AvgPool2d: 3-6                                   [1, 256, 128, 128]        --
# ├─Conv2d: 1-3                                                [1, 512, 128, 128]        131,584
# ├─Sequential: 1-4                                            [1, 32, 16, 128, 128]     --
# │    └─ResBlock3d: 2-6                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-7                     [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-8                                      [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-9                     [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-10                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-7                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-11                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-12                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-13                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-14                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-8                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-15                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-16                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-17                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-18                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-9                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-19                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-20                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-21                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-22                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-10                                      [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-23                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-24                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-25                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-26                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-11                                      [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-27                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-28                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-29                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-30                                     [1, 32, 16, 128, 128]     27,680
# ├─DenseMotionNetwork: 1-5                                    [1, 1, 128, 128]          --
# │    └─Conv3d: 2-12                                          [1, 4, 16, 128, 128]      132
# │    └─SynchronizedBatchNorm3d: 2-13                         [1, 4, 16, 128, 128]      8
# │    └─Hourglass: 2-14                                       [1, 112, 16, 128, 128]    --
# │    │    └─Encoder: 3-31                                    [1, 80, 16, 128, 128]     18,944,832
# │    │    └─Decoder: 3-32                                    [1, 112, 16, 128, 128]    23,898,096
# │    └─Conv3d: 2-15                                          [1, 16, 16, 128, 128]     614,672
# │    └─Conv2d: 2-16                                          [1, 1, 128, 128]          87,809
# ├─SameBlock2d: 1-6                                           [1, 256, 128, 128]        --
# │    └─Conv2d: 2-17                                          [1, 256, 128, 128]        1,179,904
# │    └─SynchronizedBatchNorm2d: 2-18                         [1, 256, 128, 128]        512
# │    └─LeakyReLU: 2-19                                       [1, 256, 128, 128]        --
# ├─Conv2d: 1-7                                                [1, 256, 128, 128]        65,792
# ├─Sequential: 1-8                                            [1, 256, 128, 128]        --
# │    └─ResBlock2d: 2-20                                      [1, 256, 128, 128]        --
# │    │    └─SynchronizedBatchNorm2d: 3-33                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-34                                     [1, 256, 128, 128]        590,080
# │    │    └─SynchronizedBatchNorm2d: 3-35                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-36                                     [1, 256, 128, 128]        590,080
# │    └─ResBlock2d: 2-21                                      [1, 256, 128, 128]        --
# │    │    └─SynchronizedBatchNorm2d: 3-37                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-38                                     [1, 256, 128, 128]        590,080
# │    │    └─SynchronizedBatchNorm2d: 3-39                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-40                                     [1, 256, 128, 128]        590,080
# │    └─ResBlock2d: 2-22                                      [1, 256, 128, 128]        --
# │    │    └─SynchronizedBatchNorm2d: 3-41                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-42                                     [1, 256, 128, 128]        590,080
# │    │    └─SynchronizedBatchNorm2d: 3-43                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-44                                     [1, 256, 128, 128]        590,080
# │    └─ResBlock2d: 2-23                                      [1, 256, 128, 128]        --
# │    │    └─SynchronizedBatchNorm2d: 3-45                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-46                                     [1, 256, 128, 128]        590,080
# │    │    └─SynchronizedBatchNorm2d: 3-47                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-48                                     [1, 256, 128, 128]        590,080
# │    └─ResBlock2d: 2-24                                      [1, 256, 128, 128]        --
# │    │    └─SynchronizedBatchNorm2d: 3-49                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-50                                     [1, 256, 128, 128]        590,080
# │    │    └─SynchronizedBatchNorm2d: 3-51                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-52                                     [1, 256, 128, 128]        590,080
# │    └─ResBlock2d: 2-25                                      [1, 256, 128, 128]        --
# │    │    └─SynchronizedBatchNorm2d: 3-53                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-54                                     [1, 256, 128, 128]        590,080
# │    │    └─SynchronizedBatchNorm2d: 3-55                    [1, 256, 128, 128]        512
# │    │    └─Conv2d: 3-56                                     [1, 256, 128, 128]        590,080
# ├─ModuleList: 1-9                                            --                        --
# │    └─UpBlock2d: 2-26                                       [1, 128, 256, 256]        --
# │    │    └─Conv2d: 3-57                                     [1, 128, 256, 256]        295,040
# │    │    └─SynchronizedBatchNorm2d: 3-58                    [1, 128, 256, 256]        256
# │    └─UpBlock2d: 2-27                                       [1, 64, 512, 512]         --
# │    │    └─Conv2d: 3-59                                     [1, 64, 512, 512]         73,792
# │    │    └─SynchronizedBatchNorm2d: 3-60                    [1, 64, 512, 512]         128
# ├─Conv2d: 1-10                                               [1, 3, 512, 512]          9,411
# ==============================================================================================================
# Total params: 53,101,392
# Trainable params: 53,101,392
# Non-trainable params: 0
# Total mult-adds (G): 784.14
# ==============================================================================================================
# Input size (MB): 3.15
# Forward/backward pass size (MB): 5366.74
# Params size (MB): 212.41
# Estimated Total Size (MB): 5582.29
# ==============================================================================================================
    
# OcclusionAwareSPADEGenerator
# ==============================================================================================================
# Layer (type:depth-idx)                                       Output Shape              Param #
# ==============================================================================================================
# OcclusionAwareSPADEGenerator                                 [1, 3, 512, 512]          --
# ├─SameBlock2d: 1-1                                           [1, 64, 512, 512]         --
# │    └─Conv2d: 2-1                                           [1, 64, 512, 512]         1,792
# │    └─SynchronizedBatchNorm2d: 2-2                          [1, 64, 512, 512]         128
# │    └─ReLU: 2-3                                             [1, 64, 512, 512]         --
# ├─ModuleList: 1-2                                            --                        --
# │    └─DownBlock2d: 2-4                                      [1, 128, 256, 256]        --
# │    │    └─Conv2d: 3-1                                      [1, 128, 512, 512]        73,856
# │    │    └─SynchronizedBatchNorm2d: 3-2                     [1, 128, 512, 512]        256
# │    │    └─AvgPool2d: 3-3                                   [1, 128, 256, 256]        --
# │    └─DownBlock2d: 2-5                                      [1, 256, 128, 128]        --
# │    │    └─Conv2d: 3-4                                      [1, 256, 256, 256]        295,168
# │    │    └─SynchronizedBatchNorm2d: 3-5                     [1, 256, 256, 256]        512
# │    │    └─AvgPool2d: 3-6                                   [1, 256, 128, 128]        --
# ├─Conv2d: 1-3                                                [1, 512, 128, 128]        131,584
# ├─Sequential: 1-4                                            [1, 32, 16, 128, 128]     --
# │    └─ResBlock3d: 2-6                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-7                     [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-8                                      [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-9                     [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-10                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-7                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-11                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-12                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-13                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-14                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-8                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-15                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-16                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-17                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-18                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-9                                       [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-19                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-20                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-21                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-22                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-10                                      [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-23                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-24                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-25                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-26                                     [1, 32, 16, 128, 128]     27,680
# │    └─ResBlock3d: 2-11                                      [1, 32, 16, 128, 128]     --
# │    │    └─SynchronizedBatchNorm3d: 3-27                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-28                                     [1, 32, 16, 128, 128]     27,680
# │    │    └─SynchronizedBatchNorm3d: 3-29                    [1, 32, 16, 128, 128]     64
# │    │    └─Conv3d: 3-30                                     [1, 32, 16, 128, 128]     27,680
# ├─DenseMotionNetwork: 1-5                                    [1, 1, 128, 128]          --
# │    └─Conv3d: 2-12                                          [1, 4, 16, 128, 128]      132
# │    └─SynchronizedBatchNorm3d: 2-13                         [1, 4, 16, 128, 128]      8
# │    └─Hourglass: 2-14                                       [1, 112, 16, 128, 128]    --
# │    │    └─Encoder: 3-31                                    [1, 80, 16, 128, 128]     18,944,832
# │    │    └─Decoder: 3-32                                    [1, 112, 16, 128, 128]    23,898,096
# │    └─Conv3d: 2-15                                          [1, 16, 16, 128, 128]     614,672
# │    └─Conv2d: 2-16                                          [1, 1, 128, 128]          87,809
# ├─SameBlock2d: 1-6                                           [1, 256, 128, 128]        --
# │    └─Conv2d: 2-17                                          [1, 256, 128, 128]        1,179,904
# │    └─SynchronizedBatchNorm2d: 2-18                         [1, 256, 128, 128]        512
# │    └─LeakyReLU: 2-19                                       [1, 256, 128, 128]        --
# ├─Conv2d: 1-7                                                [1, 256, 128, 128]        65,792
# ├─SPADEDecoder: 1-8                                          [1, 3, 512, 512]          --
# │    └─Conv2d: 2-20                                          [1, 512, 128, 128]        1,180,160
# │    └─SPADEResnetBlock: 2-21                                [1, 512, 128, 128]        --
# │    │    └─SPADE: 3-33                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-34                                     [1, 512, 128, 128]        2,359,808
# │    │    └─SPADE: 3-35                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-36                                     [1, 512, 128, 128]        2,359,808
# │    └─SPADEResnetBlock: 2-22                                [1, 512, 128, 128]        --
# │    │    └─SPADE: 3-37                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-38                                     [1, 512, 128, 128]        2,359,808
# │    │    └─SPADE: 3-39                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-40                                     [1, 512, 128, 128]        2,359,808
# │    └─SPADEResnetBlock: 2-23                                [1, 512, 128, 128]        --
# │    │    └─SPADE: 3-41                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-42                                     [1, 512, 128, 128]        2,359,808
# │    │    └─SPADE: 3-43                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-44                                     [1, 512, 128, 128]        2,359,808
# │    └─SPADEResnetBlock: 2-24                                [1, 512, 128, 128]        --
# │    │    └─SPADE: 3-45                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-46                                     [1, 512, 128, 128]        2,359,808
# │    │    └─SPADE: 3-47                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-48                                     [1, 512, 128, 128]        2,359,808
# │    └─SPADEResnetBlock: 2-25                                [1, 512, 128, 128]        --
# │    │    └─SPADE: 3-49                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-50                                     [1, 512, 128, 128]        2,359,808
# │    │    └─SPADE: 3-51                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-52                                     [1, 512, 128, 128]        2,359,808
# │    └─SPADEResnetBlock: 2-26                                [1, 512, 128, 128]        --
# │    │    └─SPADE: 3-53                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-54                                     [1, 512, 128, 128]        2,359,808
# │    │    └─SPADE: 3-55                                      [1, 512, 128, 128]        1,475,712
# │    │    └─Conv2d: 3-56                                     [1, 512, 128, 128]        2,359,808
# │    └─Upsample: 2-27                                        [1, 512, 256, 256]        --
# │    └─SPADEResnetBlock: 2-28                                [1, 256, 256, 256]        --
# │    │    └─SPADE: 3-57                                      [1, 512, 256, 256]        1,475,712
# │    │    └─Conv2d: 3-58                                     [1, 256, 256, 256]        131,072
# │    │    └─SPADE: 3-59                                      [1, 512, 256, 256]        1,475,712
# │    │    └─Conv2d: 3-60                                     [1, 256, 256, 256]        1,179,904
# │    │    └─SPADE: 3-61                                      [1, 256, 256, 256]        885,376
# │    │    └─Conv2d: 3-62                                     [1, 256, 256, 256]        590,080
# │    └─Upsample: 2-29                                        [1, 256, 512, 512]        --
# │    └─SPADEResnetBlock: 2-30                                [1, 64, 512, 512]         --
# │    │    └─SPADE: 3-63                                      [1, 256, 512, 512]        885,376
# │    │    └─Conv2d: 3-64                                     [1, 64, 512, 512]         16,384
# │    │    └─SPADE: 3-65                                      [1, 256, 512, 512]        885,376
# │    │    └─Conv2d: 3-66                                     [1, 64, 512, 512]         147,520
# │    │    └─SPADE: 3-67                                      [1, 64, 512, 512]         442,624
# │    │    └─Conv2d: 3-68                                     [1, 64, 512, 512]         36,928
# │    └─Conv2d: 2-31                                          [1, 3, 512, 512]          1,731
# ==============================================================================================================
# Total params: 100,988,176
# Trainable params: 100,988,176
# Non-trainable params: 0
# Total mult-adds (T): 1.78
# ==============================================================================================================
# Input size (MB): 3.15
# Forward/backward pass size (MB): 12413.17
# Params size (MB): 403.95
# Estimated Total Size (MB): 12820.27
# ==============================================================================================================

# SPADEDecoder
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# SPADEDecoder                             [1, 3, 512, 512]          --
# ├─Conv2d: 1-1                            [1, 512, 128, 128]        1,180,160
# ├─SPADEResnetBlock: 1-2                  [1, 512, 128, 128]        --
# │    └─SPADE: 2-1                        [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-1          [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-2              [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-3                  [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-4                  [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-2                       [1, 512, 128, 128]        2,359,808
# │    └─SPADE: 2-3                        [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-5          [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-6              [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-7                  [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-8                  [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-4                       [1, 512, 128, 128]        2,359,808
# ├─SPADEResnetBlock: 1-3                  [1, 512, 128, 128]        --
# │    └─SPADE: 2-5                        [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-9          [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-10             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-11                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-12                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-6                       [1, 512, 128, 128]        2,359,808
# │    └─SPADE: 2-7                        [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-13         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-14             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-15                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-16                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-8                       [1, 512, 128, 128]        2,359,808
# ├─SPADEResnetBlock: 1-4                  [1, 512, 128, 128]        --
# │    └─SPADE: 2-9                        [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-17         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-18             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-19                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-20                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-10                      [1, 512, 128, 128]        2,359,808
# │    └─SPADE: 2-11                       [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-21         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-22             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-23                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-24                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-12                      [1, 512, 128, 128]        2,359,808
# ├─SPADEResnetBlock: 1-5                  [1, 512, 128, 128]        --
# │    └─SPADE: 2-13                       [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-25         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-26             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-27                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-28                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-14                      [1, 512, 128, 128]        2,359,808
# │    └─SPADE: 2-15                       [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-29         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-30             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-31                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-32                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-16                      [1, 512, 128, 128]        2,359,808
# ├─SPADEResnetBlock: 1-6                  [1, 512, 128, 128]        --
# │    └─SPADE: 2-17                       [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-33         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-34             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-35                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-36                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-18                      [1, 512, 128, 128]        2,359,808
# │    └─SPADE: 2-19                       [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-37         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-38             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-39                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-40                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-20                      [1, 512, 128, 128]        2,359,808
# ├─SPADEResnetBlock: 1-7                  [1, 512, 128, 128]        --
# │    └─SPADE: 2-21                       [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-41         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-42             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-43                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-44                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-22                      [1, 512, 128, 128]        2,359,808
# │    └─SPADE: 2-23                       [1, 512, 128, 128]        --
# │    │    └─InstanceNorm2d: 3-45         [1, 512, 128, 128]        --
# │    │    └─Sequential: 3-46             [1, 128, 128, 128]        295,040
# │    │    └─Conv2d: 3-47                 [1, 512, 128, 128]        590,336
# │    │    └─Conv2d: 3-48                 [1, 512, 128, 128]        590,336
# │    └─Conv2d: 2-24                      [1, 512, 128, 128]        2,359,808
# ├─Upsample: 1-8                          [1, 512, 256, 256]        --
# ├─SPADEResnetBlock: 1-9                  [1, 256, 256, 256]        --
# │    └─SPADE: 2-25                       [1, 512, 256, 256]        --
# │    │    └─InstanceNorm2d: 3-49         [1, 512, 256, 256]        --
# │    │    └─Sequential: 3-50             [1, 128, 256, 256]        295,040
# │    │    └─Conv2d: 3-51                 [1, 512, 256, 256]        590,336
# │    │    └─Conv2d: 3-52                 [1, 512, 256, 256]        590,336
# │    └─Conv2d: 2-26                      [1, 256, 256, 256]        131,072
# │    └─SPADE: 2-27                       [1, 512, 256, 256]        --
# │    │    └─InstanceNorm2d: 3-53         [1, 512, 256, 256]        --
# │    │    └─Sequential: 3-54             [1, 128, 256, 256]        295,040
# │    │    └─Conv2d: 3-55                 [1, 512, 256, 256]        590,336
# │    │    └─Conv2d: 3-56                 [1, 512, 256, 256]        590,336
# │    └─Conv2d: 2-28                      [1, 256, 256, 256]        1,179,904
# │    └─SPADE: 2-29                       [1, 256, 256, 256]        --
# │    │    └─InstanceNorm2d: 3-57         [1, 256, 256, 256]        --
# │    │    └─Sequential: 3-58             [1, 128, 256, 256]        295,040
# │    │    └─Conv2d: 3-59                 [1, 256, 256, 256]        295,168
# │    │    └─Conv2d: 3-60                 [1, 256, 256, 256]        295,168
# │    └─Conv2d: 2-30                      [1, 256, 256, 256]        590,080
# ├─Upsample: 1-10                         [1, 256, 512, 512]        --
# ├─SPADEResnetBlock: 1-11                 [1, 64, 512, 512]         --
# │    └─SPADE: 2-31                       [1, 256, 512, 512]        --
# │    │    └─InstanceNorm2d: 3-61         [1, 256, 512, 512]        --
# │    │    └─Sequential: 3-62             [1, 128, 512, 512]        295,040
# │    │    └─Conv2d: 3-63                 [1, 256, 512, 512]        295,168
# │    │    └─Conv2d: 3-64                 [1, 256, 512, 512]        295,168
# │    └─Conv2d: 2-32                      [1, 64, 512, 512]         16,384
# │    └─SPADE: 2-33                       [1, 256, 512, 512]        --
# │    │    └─InstanceNorm2d: 3-65         [1, 256, 512, 512]        --
# │    │    └─Sequential: 3-66             [1, 128, 512, 512]        295,040
# │    │    └─Conv2d: 3-67                 [1, 256, 512, 512]        295,168
# │    │    └─Conv2d: 3-68                 [1, 256, 512, 512]        295,168
# │    └─Conv2d: 2-34                      [1, 64, 512, 512]         147,520
# │    └─SPADE: 2-35                       [1, 64, 512, 512]         --
# │    │    └─InstanceNorm2d: 3-69         [1, 64, 512, 512]         --
# │    │    └─Sequential: 3-70             [1, 128, 512, 512]        295,040
# │    │    └─Conv2d: 3-71                 [1, 64, 512, 512]         73,792
# │    │    └─Conv2d: 3-72                 [1, 64, 512, 512]         73,792
# │    └─Conv2d: 2-36                      [1, 64, 512, 512]         36,928
# ├─Conv2d: 1-12                           [1, 3, 512, 512]          1,731
# ==========================================================================================
# Total params: 55,360,195
# Trainable params: 55,360,195
# Non-trainable params: 0
# Total mult-adds (T): 1.16
# ==========================================================================================
# Input size (MB): 16.78
# Forward/backward pass size (MB): 8260.68
# Params size (MB): 221.44
# Estimated Total Size (MB): 8498.90
# ==========================================================================================




# 100,988,176 - 43,545,549 - 55,360,195 = 2,082,432
#               densemotion   spade

# 2,082,432 + 43,545,549 = 45,627,981
