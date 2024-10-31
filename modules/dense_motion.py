from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, kp2gaussian

from modules.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)

 
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)
        
  
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
   
        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=False)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]

        heatmap_coordinate_grid = make_coordinate_grid(spatial_size, kp_driving['value'].type())
      
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01, coordinate_grid=heatmap_coordinate_grid)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01, coordinate_grid=heatmap_coordinate_grid)
        heatmap = gaussian_driving - gaussian_source
        heatmap_zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([heatmap_zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature) # conv3d
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        input = torch.cat([heatmap, deformed_feature], dim=2)
        input = input.view(bs, -1, d, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction) # conv3d
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict

# DenseMotionNetwork
# =========================================================================================================
# Layer (type:depth-idx)                                  Output Shape              Param #
# =========================================================================================================
# DenseMotionNetwork                                      [1, 1, 128, 128]          --
# ├─Conv3d: 1-1                                           [1, 4, 16, 128, 128]      132
# ├─SynchronizedBatchNorm3d: 1-2                          [1, 4, 16, 128, 128]      8
# ├─Hourglass: 1-3                                        [1, 112, 16, 128, 128]    --
# │    └─Encoder: 2-1                                     [1, 80, 16, 128, 128]     --
# │    │    └─ModuleList: 3-1                             --                        18,944,832
# │    └─Decoder: 2-2                                     [1, 112, 16, 128, 128]    --
# │    │    └─ModuleList: 3-2                             --                        23,559,072
# │    │    └─Conv3d: 3-3                                 [1, 112, 16, 128, 128]    338,800
# │    │    └─SynchronizedBatchNorm3d: 3-4                [1, 112, 16, 128, 128]    224
# ├─Conv3d: 1-4                                           [1, 16, 16, 128, 128]     614,672
# ├─Conv2d: 1-5                                           [1, 1, 128, 128]          87,809
# =========================================================================================================
# Total params: 43,545,549
# Trainable params: 43,545,549
# Non-trainable params: 0
# Total mult-adds (G): 476.15
# =========================================================================================================
# Input size (MB): 33.55
# Forward/backward pass size (MB): 1300.37
# Params size (MB): 174.18
# Estimated Total Size (MB): 1508.10
# =========================================================================================================