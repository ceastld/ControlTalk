import os
import cv2
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.discriminator import MultiScaleDiscriminator
from modules.model import ImagePyramide

def load_checkpoints(config_path, checkpoint_path, gen="spade", device='cuda'):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    
    if os.path.isfile(checkpoint_path):
        # checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        he_estimator.load_state_dict(checkpoint['he_estimator'])
        print("generator & kp_detector & he_estimator & loaded from: ", checkpoint_path)
    generator.eval()
    kp_detector.eval()
    he_estimator.eval()

    discriminator = None
    
    generator.to(device)
    kp_detector.to(device)
    he_estimator.to(device)
    return generator, kp_detector, he_estimator, discriminator, config['train_params']


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=-1)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)
    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=True, free_view=False, yaw=0, pitch=0, roll=0):
    device = kp_canonical['value'].device
    kp = kp_canonical['value']
    if not free_view:
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)
    else:
        if yaw is not None:
            yaw = torch.tensor([yaw]).to(device)
        else:
            yaw = he['yaw']
            yaw = headpose_pred_to_degree(yaw)
        if pitch is not None:
            pitch = torch.tensor([pitch]).to(device)
        else:
            pitch = he['pitch']
            pitch = headpose_pred_to_degree(pitch)
        if roll is not None:
            roll = torch.tensor([roll]).to(device)
        else:
            roll = he['roll']
            roll = headpose_pred_to_degree(roll)

    t, exp = he['t'], he['exp']
    rot_mat = get_rotation_matrix(yaw, pitch, roll)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
    t = t.clone().unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.clone().view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}


class Renderer(torch.nn.Module):
    def __init__(self, config_path, checkpoint_path, gen="spade", device='cuda', pic_size=256):
        super(Renderer, self).__init__()
        self.generator, self.kp_detector, self.he_estimator, self.discriminator, train_params = load_checkpoints(config_path, checkpoint_path, gen, device=device)

        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, 3)

        self.device = device
        self.pic_size = pic_size

    def forward(self, source, driving, estimate_jacobian=False, extract_driving_only=False, no_need_drv_kp=False):

        if extract_driving_only:
            if True:
                driving = driving.to(self.device)
                if no_need_drv_kp:
                    kp_canical_driving = None
                else:
                    kp_canical_driving = self.kp_detector(driving)
                he_driving = self.he_estimator(driving)
                return he_driving, kp_canical_driving

        if True:
            source = source.to(self.device)
            driving = driving.to(self.device)
            kp_canonical = self.kp_detector(source)
            he_source = self.he_estimator(source)
            he_driving = self.he_estimator(driving)
        

        kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
        kp_driving = keypoint_transformation(kp_canonical, he_driving, estimate_jacobian)
        out = self.generator(source, kp_source=kp_source, kp_driving=kp_driving)

        return out


    def forward_kp_he(self, source, kp_canonical, he_source, he_driving, estimate_jacobian=False):

        if isinstance(source, torch.Tensor) and source.shape[-1] == self.pic_size:
            pass
        elif (isinstance(source, list) or isinstance(source, tuple)) and isinstance(source[0], str): # 文件路径
            source_tmp = []
            for souce_im_path in source:
                source_im = cv2.imread(souce_im_path)
                if source_im.shape[0] != self.pic_size or source_im.shape[1] != self.pic_size:
                    source_im = cv2.resize(source_im, (self.pic_size, self.pic_size))
           
                source_im = cv2.cvtColor(source_im, cv2.COLOR_BGR2RGB)
                source_im = source_im / 255.
                source_tmp.append(source_im)
            source = np.array(source_tmp)
            source_tmp = []
            source = torch.tensor(source, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)


        kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
        kp_driving = keypoint_transformation(kp_canonical, he_driving, estimate_jacobian)


        if True:
            out = self.generator(source, kp_source=kp_source, kp_driving=kp_driving)
        return out

    def forward_img_2_vid(self, source_img, he_source, he_driving_list, num_frames, estimate_jacobian=False):

        if True:
            kp_canonical_source = self.kp_detector(source_img)
        kp_source = keypoint_transformation(kp_canonical_source, he_source, estimate_jacobian)
        predictions = []
        for i in tqdm(range(num_frames), "Render:"):
            kp_driving_single = keypoint_transformation(kp_canonical_source, he_driving_list[i], estimate_jacobian)
            if True:
                out = self.generator(source_img, kp_source=kp_source, kp_driving=kp_driving_single)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions


if __name__ == "__main__":
    pass