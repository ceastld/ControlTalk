from torch import nn
import torch.nn.functional as F
# from modules.util import kp2gaussian
import torch


class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out

class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, **kwargs):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks): # 4
            down_blocks.append(
                DownBlock2d(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):

        # mouth
        # bs, _, h, w = x.shape
        # x = x[:, :, h // 2 - h//8 : h // 2 + (h*3)//8, w // 2 - w // 4 : w // 2 + w // 4]
        # x = x[:, :, h // 2 - h * 0//8 : h // 2 + (h*4)//8, w // 2 - w // 4 : w // 2 + w // 4]
        # x = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear')

        feature_maps = []
        out = x

        for down_block in self.down_blocks:
            feature = down_block(out)
            feature_maps.append(feature)
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map



class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            # print("key: ", key, x[key].shape)
            feature_maps, prediction_map = disc(x[key])
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict

if __name__ == "__main__":
    pass
# 256 input
# key:  prediction_1
# feature_maps:  torch.Size([1, 128, 126, 126])
# feature_maps:  torch.Size([1, 256, 61, 61])
# feature_maps:  torch.Size([1, 512, 29, 29])
# feature_maps:  torch.Size([1, 512, 26, 26])
# out:  torch.Size([1, 512, 26, 26])
# 
# key:  prediction_0.5
# feature_maps:  torch.Size([1, 128, 62, 62])
# feature_maps:  torch.Size([1, 256, 29, 29])
# feature_maps:  torch.Size([1, 512, 13, 13])
# feature_maps:  torch.Size([1, 512, 10, 10])
# out:  torch.Size([1, 512, 10, 10])
# 
# key:  prediction_0.25
# feature_maps:  torch.Size([1, 128, 30, 30])
# feature_maps:  torch.Size([1, 256, 13, 13])
# feature_maps:  torch.Size([1, 512, 5, 5])
# feature_maps:  torch.Size([1, 512, 2, 2])
# out:  torch.Size([1, 512, 2, 2])
# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# MultiScaleDiscriminator                            [1, 1, 2, 2]              --
# ├─ModuleDict: 1-1                                  --                        --
# │    └─Discriminator: 2-1                          [1, 128, 126, 126]        --
# │    │    └─ModuleList: 3-1                        --                        6,825,856
# │    │    └─Conv2d: 3-2                            [1, 1, 26, 26]            513
# │    └─Discriminator: 2-2                          [1, 128, 62, 62]          --
# │    │    └─ModuleList: 3-3                        --                        6,825,856
# │    │    └─Conv2d: 3-4                            [1, 1, 10, 10]            513
# │    └─Discriminator: 2-3                          [1, 128, 30, 30]          --
# │    │    └─ModuleList: 3-5                        --                        6,825,856
# │    │    └─Conv2d: 3-6                            [1, 1, 2, 2]              513
# ====================================================================================================
# Total params: 20,479,107
# Trainable params: 20,479,107
# Non-trainable params: 0
# Total mult-adds (G): 22.62
# ====================================================================================================
# Input size (MB): 1.03
# Forward/backward pass size (MB): 204.88
# Params size (MB): 81.92
# Estimated Total Size (MB): 287.83
# ====================================================================================================


# 512 input
# key:  prediction_1
# feature_maps:  torch.Size([1, 128, 254, 254])
# feature_maps:  torch.Size([1, 256, 125, 125])
# feature_maps:  torch.Size([1, 512, 61, 61])
# feature_maps:  torch.Size([1, 512, 58, 58])
# out:  torch.Size([1, 512, 58, 58])
# key:  prediction_0.5
# feature_maps:  torch.Size([1, 128, 126, 126])
# feature_maps:  torch.Size([1, 256, 61, 61])
# feature_maps:  torch.Size([1, 512, 29, 29])
# feature_maps:  torch.Size([1, 512, 26, 26])
# out:  torch.Size([1, 512, 26, 26])
# key:  prediction_0.25
# feature_maps:  torch.Size([1, 128, 62, 62])
# feature_maps:  torch.Size([1, 256, 29, 29])
# feature_maps:  torch.Size([1, 512, 13, 13])
# feature_maps:  torch.Size([1, 512, 10, 10])
# out:  torch.Size([1, 512, 10, 10])
# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# MultiScaleDiscriminator                            [1, 1, 10, 10]            --
# ├─ModuleDict: 1-1                                  --                        --
# │    └─Discriminator: 2-1                          [1, 128, 254, 254]        --
# │    │    └─ModuleList: 3-1                        --                        6,825,856
# │    │    └─Conv2d: 3-2                            [1, 1, 58, 58]            513
# │    └─Discriminator: 2-2                          [1, 128, 126, 126]        --
# │    │    └─ModuleList: 3-3                        --                        6,825,856
# │    │    └─Conv2d: 3-4                            [1, 1, 26, 26]            513
# │    └─Discriminator: 2-3                          [1, 128, 62, 62]          --
# │    │    └─ModuleList: 3-5                        --                        6,825,856
# │    │    └─Conv2d: 3-6                            [1, 1, 10, 10]            513
# ====================================================================================================



# Discriminator(
#   (down_blocks): ModuleList(
#     (0): DownBlock2d(
#       (conv): Conv2d(3, 64, kernel_size=(4, 4), stride=(1, 1))
#     )
#     (1): DownBlock2d(
#       (conv): Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1))
#       (norm): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
#     )
#     (2): DownBlock2d(
#       (conv): Conv2d(128, 256, kernel_size=(4, 4), stride=(1, 1))
#       (norm): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
#     )
#     (3): DownBlock2d(
#       (conv): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1))
#       (norm): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
#     )
#   )
#   (conv): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
# )
# feature_maps:  torch.Size([1, 64, 254, 254])
# feature_maps:  torch.Size([1, 128, 125, 125])
# feature_maps:  torch.Size([1, 256, 61, 61])
# feature_maps:  torch.Size([1, 512, 58, 58])
# out:  torch.Size([1, 512, 58, 58])
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# Discriminator                            [1, 64, 254, 254]         --
# ├─ModuleList: 1-1                        --                        --
# │    └─DownBlock2d: 2-1                  [1, 64, 254, 254]         --
# │    │    └─Conv2d: 3-1                  [1, 64, 509, 509]         3,136
# │    └─DownBlock2d: 2-2                  [1, 128, 125, 125]        --
# │    │    └─Conv2d: 3-2                  [1, 128, 251, 251]        131,200
# │    │    └─InstanceNorm2d: 3-3          [1, 128, 251, 251]        256
# │    └─DownBlock2d: 2-3                  [1, 256, 61, 61]          --
# │    │    └─Conv2d: 3-4                  [1, 256, 122, 122]        524,544
# │    │    └─InstanceNorm2d: 3-5          [1, 256, 122, 122]        512
# │    └─DownBlock2d: 2-4                  [1, 512, 58, 58]          --
# │    │    └─Conv2d: 3-6                  [1, 512, 58, 58]          2,097,664
# │    │    └─InstanceNorm2d: 3-7          [1, 512, 58, 58]          1,024
# ├─Conv2d: 1-2                            [1, 1, 58, 58]            513
# ==========================================================================================
# Total params: 2,758,849
# Trainable params: 2,758,849
# Non-trainable params: 0
# Total mult-adds (G): 23.94
# ==========================================================================================
# Input size (MB): 3.15
# Forward/backward pass size (MB): 350.23
# Params size (MB): 11.04
# Estimated Total Size (MB): 364.41
# ==========================================================================================