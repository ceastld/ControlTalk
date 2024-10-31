from torch import nn
import torch
import torch.nn.functional as F

from modules.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from modules.util import KPHourglass, make_coordinate_grid, AntiAliasInterpolation2d, ResBottleneck

class KPDetector(nn.Module):
    """
    Detecting canonical keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, reshape_channel, reshape_depth,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1, single_jacobian_map=False):
        super(KPDetector, self).__init__()

        self.predictor = KPHourglass(block_expansion, in_features=image_channel,
                                     max_features=max_features,  reshape_features=reshape_channel, reshape_depth=reshape_depth, num_blocks=num_blocks)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=3, padding=1)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=9 * self.num_jacobian_maps, kernel_size=3, padding=1)
            '''
            initial as:
            [[1 0 0]
             [0 1 0]
             [0 0 1]]
            '''
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type(), device=heatmap.device).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3, 4))
        kp = {'value': value}
        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        feature_map = self.predictor(x)
        prediction = self.kp(feature_map) # conv3d

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 9, final_shape[2],
                                                final_shape[3], final_shape[4])
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 9, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 3, 3)
            out['jacobian'] = jacobian

        return out


class HEEstimator(nn.Module):
    """
    Estimating head pose and expression.
    """

    def __init__(self, block_expansion, feature_channel, num_kp, image_channel, max_features, num_bins=66, estimate_jacobian=True):
        super(HEEstimator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=image_channel, out_channels=block_expansion, kernel_size=7, padding=3, stride=2)
        self.norm1 = BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=block_expansion, out_channels=256, kernel_size=1)
        self.norm2 = BatchNorm2d(256, affine=True)

        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module('b1_'+ str(i), ResBottleneck(in_features=256, stride=1))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.norm3 = BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)

        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module('b3_'+ str(i), ResBottleneck(in_features=512, stride=1))

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.norm4 = BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)

        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module('b5_'+ str(i), ResBottleneck(in_features=1024, stride=1))

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.norm5 = BatchNorm2d(2048, affine=True)
        self.block6 = ResBottleneck(in_features=2048, stride=2)

        self.block7 = nn.Sequential()
        for i in range(2):
            self.block7.add_module('b7_'+ str(i), ResBottleneck(in_features=2048, stride=1))

        self.fc_roll = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)
        self.fc_yaw = nn.Linear(2048, num_bins)

        self.fc_t = nn.Linear(2048, 3)

        self.fc_exp = nn.Linear(2048, 3*num_kp)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.block2(out)

        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.block4(out)

        out = self.block5(out)

        out = self.conv5(out)
        out = self.norm5(out)
        out = F.relu(out)
        out = self.block6(out)

        out = self.block7(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.shape[0], -1)

        yaw = self.fc_roll(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_yaw(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)


        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 
                't': t, 
                'exp': exp}

if __name__ == "__main__":
   pass
# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# KPDetector                                         [1, 15, 3]                --
# ├─KPHourglass: 1-1                                 [1, 32, 16, 512, 512]     --
# │    └─Sequential: 2-1                             [1, 1024, 16, 16]         --
# │    │    └─DownBlock2d: 3-1                       [1, 64, 256, 256]         1,920
# │    │    └─DownBlock2d: 3-2                       [1, 128, 128, 128]        74,112
# │    │    └─DownBlock2d: 3-3                       [1, 256, 64, 64]          295,680
# │    │    └─DownBlock2d: 3-4                       [1, 512, 32, 32]          1,181,184
# │    │    └─DownBlock2d: 3-5                       [1, 1024, 16, 16]         4,721,664
# │    └─Conv2d: 2-2                                 [1, 16384, 16, 16]        16,793,600
# │    └─Sequential: 2-3                             [1, 32, 16, 512, 512]     --
# │    │    └─UpBlock3d: 3-6                         [1, 512, 16, 32, 32]      14,157,312
# │    │    └─UpBlock3d: 3-7                         [1, 256, 16, 64, 64]      3,539,712
# │    │    └─UpBlock3d: 3-8                         [1, 128, 16, 128, 128]    885,120
# │    │    └─UpBlock3d: 3-9                         [1, 64, 16, 256, 256]     221,376
# │    │    └─UpBlock3d: 3-10                        [1, 32, 16, 512, 512]     55,392
# ├─Conv3d: 1-2                                      [1, 15, 16, 512, 512]     12,975
# ====================================================================================================
# Total params: 41,940,047
# Trainable params: 41,940,047
# Non-trainable params: 0
# Total mult-adds (T): 1.24
# ====================================================================================================
# Input size (MB): 3.15
# Forward/backward pass size (MB): 5217.71
# Params size (MB): 167.76
# Estimated Total Size (MB): 5388.62
# ====================================================================================================

# HEEstimator
# ===============================================================================================
# Layer (type:depth-idx)                        Output Shape              Param #
# ===============================================================================================
# HEEstimator                                   [1, 45]                   --
# ├─Conv2d: 1-1                                 [1, 64, 256, 256]         9,472
# ├─SynchronizedBatchNorm2d: 1-2                [1, 64, 256, 256]         128
# ├─MaxPool2d: 1-3                              [1, 64, 128, 128]         --
# ├─Conv2d: 1-4                                 [1, 256, 128, 128]        16,640
# ├─SynchronizedBatchNorm2d: 1-5                [1, 256, 128, 128]        512
# ├─Sequential: 1-6                             [1, 256, 128, 128]        --
# │    └─ResBottleneck: 2-1                     [1, 256, 128, 128]        --
# │    │    └─Conv2d: 3-1                       [1, 64, 128, 128]         16,448
# │    │    └─SynchronizedBatchNorm2d: 3-2      [1, 64, 128, 128]         128
# │    │    └─Conv2d: 3-3                       [1, 64, 128, 128]         36,928
# │    │    └─SynchronizedBatchNorm2d: 3-4      [1, 64, 128, 128]         128
# │    │    └─Conv2d: 3-5                       [1, 256, 128, 128]        16,640
# │    │    └─SynchronizedBatchNorm2d: 3-6      [1, 256, 128, 128]        512
# │    └─ResBottleneck: 2-2                     [1, 256, 128, 128]        --
# │    │    └─Conv2d: 3-7                       [1, 64, 128, 128]         16,448
# │    │    └─SynchronizedBatchNorm2d: 3-8      [1, 64, 128, 128]         128
# │    │    └─Conv2d: 3-9                       [1, 64, 128, 128]         36,928
# │    │    └─SynchronizedBatchNorm2d: 3-10     [1, 64, 128, 128]         128
# │    │    └─Conv2d: 3-11                      [1, 256, 128, 128]        16,640
# │    │    └─SynchronizedBatchNorm2d: 3-12     [1, 256, 128, 128]        512
# │    └─ResBottleneck: 2-3                     [1, 256, 128, 128]        --
# │    │    └─Conv2d: 3-13                      [1, 64, 128, 128]         16,448
# │    │    └─SynchronizedBatchNorm2d: 3-14     [1, 64, 128, 128]         128
# │    │    └─Conv2d: 3-15                      [1, 64, 128, 128]         36,928
# │    │    └─SynchronizedBatchNorm2d: 3-16     [1, 64, 128, 128]         128
# │    │    └─Conv2d: 3-17                      [1, 256, 128, 128]        16,640
# │    │    └─SynchronizedBatchNorm2d: 3-18     [1, 256, 128, 128]        512
# ├─Conv2d: 1-7                                 [1, 512, 128, 128]        131,584
# ├─SynchronizedBatchNorm2d: 1-8                [1, 512, 128, 128]        1,024
# ├─ResBottleneck: 1-9                          [1, 512, 64, 64]          --
# │    └─Conv2d: 2-4                            [1, 128, 128, 128]        65,664
# │    └─SynchronizedBatchNorm2d: 2-5           [1, 128, 128, 128]        256
# │    └─Conv2d: 2-6                            [1, 128, 64, 64]          147,584
# │    └─SynchronizedBatchNorm2d: 2-7           [1, 128, 64, 64]          256
# │    └─Conv2d: 2-8                            [1, 512, 64, 64]          66,048
# │    └─SynchronizedBatchNorm2d: 2-9           [1, 512, 64, 64]          1,024
# │    └─Conv2d: 2-10                           [1, 512, 64, 64]          262,656
# │    └─SynchronizedBatchNorm2d: 2-11          [1, 512, 64, 64]          1,024
# ├─Sequential: 1-10                            [1, 512, 64, 64]          --
# │    └─ResBottleneck: 2-12                    [1, 512, 64, 64]          --
# │    │    └─Conv2d: 3-19                      [1, 128, 64, 64]          65,664
# │    │    └─SynchronizedBatchNorm2d: 3-20     [1, 128, 64, 64]          256
# │    │    └─Conv2d: 3-21                      [1, 128, 64, 64]          147,584
# │    │    └─SynchronizedBatchNorm2d: 3-22     [1, 128, 64, 64]          256
# │    │    └─Conv2d: 3-23                      [1, 512, 64, 64]          66,048
# │    │    └─SynchronizedBatchNorm2d: 3-24     [1, 512, 64, 64]          1,024
# │    └─ResBottleneck: 2-13                    [1, 512, 64, 64]          --
# │    │    └─Conv2d: 3-25                      [1, 128, 64, 64]          65,664
# │    │    └─SynchronizedBatchNorm2d: 3-26     [1, 128, 64, 64]          256
# │    │    └─Conv2d: 3-27                      [1, 128, 64, 64]          147,584
# │    │    └─SynchronizedBatchNorm2d: 3-28     [1, 128, 64, 64]          256
# │    │    └─Conv2d: 3-29                      [1, 512, 64, 64]          66,048
# │    │    └─SynchronizedBatchNorm2d: 3-30     [1, 512, 64, 64]          1,024
# │    └─ResBottleneck: 2-14                    [1, 512, 64, 64]          --
# │    │    └─Conv2d: 3-31                      [1, 128, 64, 64]          65,664
# │    │    └─SynchronizedBatchNorm2d: 3-32     [1, 128, 64, 64]          256
# │    │    └─Conv2d: 3-33                      [1, 128, 64, 64]          147,584
# │    │    └─SynchronizedBatchNorm2d: 3-34     [1, 128, 64, 64]          256
# │    │    └─Conv2d: 3-35                      [1, 512, 64, 64]          66,048
# │    │    └─SynchronizedBatchNorm2d: 3-36     [1, 512, 64, 64]          1,024
# ├─Conv2d: 1-11                                [1, 1024, 64, 64]         525,312
# ├─SynchronizedBatchNorm2d: 1-12               [1, 1024, 64, 64]         2,048
# ├─ResBottleneck: 1-13                         [1, 1024, 32, 32]         --
# │    └─Conv2d: 2-15                           [1, 256, 64, 64]          262,400
# │    └─SynchronizedBatchNorm2d: 2-16          [1, 256, 64, 64]          512
# │    └─Conv2d: 2-17                           [1, 256, 32, 32]          590,080
# │    └─SynchronizedBatchNorm2d: 2-18          [1, 256, 32, 32]          512
# │    └─Conv2d: 2-19                           [1, 1024, 32, 32]         263,168
# │    └─SynchronizedBatchNorm2d: 2-20          [1, 1024, 32, 32]         2,048
# │    └─Conv2d: 2-21                           [1, 1024, 32, 32]         1,049,600
# │    └─SynchronizedBatchNorm2d: 2-22          [1, 1024, 32, 32]         2,048
# ├─Sequential: 1-14                            [1, 1024, 32, 32]         --
# │    └─ResBottleneck: 2-23                    [1, 1024, 32, 32]         --
# │    │    └─Conv2d: 3-37                      [1, 256, 32, 32]          262,400
# │    │    └─SynchronizedBatchNorm2d: 3-38     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-39                      [1, 256, 32, 32]          590,080
# │    │    └─SynchronizedBatchNorm2d: 3-40     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-41                      [1, 1024, 32, 32]         263,168
# │    │    └─SynchronizedBatchNorm2d: 3-42     [1, 1024, 32, 32]         2,048
# │    └─ResBottleneck: 2-24                    [1, 1024, 32, 32]         --
# │    │    └─Conv2d: 3-43                      [1, 256, 32, 32]          262,400
# │    │    └─SynchronizedBatchNorm2d: 3-44     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-45                      [1, 256, 32, 32]          590,080
# │    │    └─SynchronizedBatchNorm2d: 3-46     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-47                      [1, 1024, 32, 32]         263,168
# │    │    └─SynchronizedBatchNorm2d: 3-48     [1, 1024, 32, 32]         2,048
# │    └─ResBottleneck: 2-25                    [1, 1024, 32, 32]         --
# │    │    └─Conv2d: 3-49                      [1, 256, 32, 32]          262,400
# │    │    └─SynchronizedBatchNorm2d: 3-50     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-51                      [1, 256, 32, 32]          590,080
# │    │    └─SynchronizedBatchNorm2d: 3-52     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-53                      [1, 1024, 32, 32]         263,168
# │    │    └─SynchronizedBatchNorm2d: 3-54     [1, 1024, 32, 32]         2,048
# │    └─ResBottleneck: 2-26                    [1, 1024, 32, 32]         --
# │    │    └─Conv2d: 3-55                      [1, 256, 32, 32]          262,400
# │    │    └─SynchronizedBatchNorm2d: 3-56     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-57                      [1, 256, 32, 32]          590,080
# │    │    └─SynchronizedBatchNorm2d: 3-58     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-59                      [1, 1024, 32, 32]         263,168
# │    │    └─SynchronizedBatchNorm2d: 3-60     [1, 1024, 32, 32]         2,048
# │    └─ResBottleneck: 2-27                    [1, 1024, 32, 32]         --
# │    │    └─Conv2d: 3-61                      [1, 256, 32, 32]          262,400
# │    │    └─SynchronizedBatchNorm2d: 3-62     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-63                      [1, 256, 32, 32]          590,080
# │    │    └─SynchronizedBatchNorm2d: 3-64     [1, 256, 32, 32]          512
# │    │    └─Conv2d: 3-65                      [1, 1024, 32, 32]         263,168
# │    │    └─SynchronizedBatchNorm2d: 3-66     [1, 1024, 32, 32]         2,048
# ├─Conv2d: 1-15                                [1, 2048, 32, 32]         2,099,200
# ├─SynchronizedBatchNorm2d: 1-16               [1, 2048, 32, 32]         4,096
# ├─ResBottleneck: 1-17                         [1, 2048, 16, 16]         --
# │    └─Conv2d: 2-28                           [1, 512, 32, 32]          1,049,088
# │    └─SynchronizedBatchNorm2d: 2-29          [1, 512, 32, 32]          1,024
# │    └─Conv2d: 2-30                           [1, 512, 16, 16]          2,359,808
# │    └─SynchronizedBatchNorm2d: 2-31          [1, 512, 16, 16]          1,024
# │    └─Conv2d: 2-32                           [1, 2048, 16, 16]         1,050,624
# │    └─SynchronizedBatchNorm2d: 2-33          [1, 2048, 16, 16]         4,096
# │    └─Conv2d: 2-34                           [1, 2048, 16, 16]         4,196,352
# │    └─SynchronizedBatchNorm2d: 2-35          [1, 2048, 16, 16]         4,096
# ├─Sequential: 1-18                            [1, 2048, 16, 16]         --
# │    └─ResBottleneck: 2-36                    [1, 2048, 16, 16]         --
# │    │    └─Conv2d: 3-67                      [1, 512, 16, 16]          1,049,088
# │    │    └─SynchronizedBatchNorm2d: 3-68     [1, 512, 16, 16]          1,024
# │    │    └─Conv2d: 3-69                      [1, 512, 16, 16]          2,359,808
# │    │    └─SynchronizedBatchNorm2d: 3-70     [1, 512, 16, 16]          1,024
# │    │    └─Conv2d: 3-71                      [1, 2048, 16, 16]         1,050,624
# │    │    └─SynchronizedBatchNorm2d: 3-72     [1, 2048, 16, 16]         4,096
# │    └─ResBottleneck: 2-37                    [1, 2048, 16, 16]         --
# │    │    └─Conv2d: 3-73                      [1, 512, 16, 16]          1,049,088
# │    │    └─SynchronizedBatchNorm2d: 3-74     [1, 512, 16, 16]          1,024
# │    │    └─Conv2d: 3-75                      [1, 512, 16, 16]          2,359,808
# │    │    └─SynchronizedBatchNorm2d: 3-76     [1, 512, 16, 16]          1,024
# │    │    └─Conv2d: 3-77                      [1, 2048, 16, 16]         1,050,624
# │    │    └─SynchronizedBatchNorm2d: 3-78     [1, 2048, 16, 16]         4,096
# ├─Linear: 1-19                                [1, 66]                   135,234
# ├─Linear: 1-20                                [1, 66]                   135,234
# ├─Linear: 1-21                                [1, 66]                   135,234
# ├─Linear: 1-22                                [1, 3]                    6,147
# ├─Linear: 1-23                                [1, 45]                   92,205
# ===============================================================================================
# Total params: 30,254,838
# Trainable params: 30,254,838
# Non-trainable params: 0
# Total mult-adds (G): 31.29
# ===============================================================================================
# Input size (MB): 3.15
# Forward/backward pass size (MB): 1163.92
# Params size (MB): 121.02
# Estimated Total Size (MB): 1288.09
# ===============================================================================================
