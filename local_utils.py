import os
import cv2, psutil
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.spatial import ConvexHull

import torch
from torchvision.transforms import Resize
from torch.nn import functional as F
from torchvision import models

from hparams import hparams as hp

def mem_usage_get():
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
    print(pid, 'memory use:', memoryUse)
    return memoryUse


def crop_tensor_mouth(img_tensor, lm, mouth_size=128, debug=False, pic_size=256, prefix=''):
    if debug:
        print("img_tensor: ", img_tensor.shape, "lm: ", lm.shape, "mouth_size: ", mouth_size)
    lm = lm.cpu().numpy()
    if img_tensor.shape[1] == 256 and  512 == pic_size:
        lm = lm / 2.0
    if debug:
        img = img_tensor.cpu().detach().numpy().transpose(1, 2, 0).copy()
        img = (img * 255).astype(np.uint8)
        print("img: ", img.shape, type(img)) # (512, 512, 3)
        for i in range(lm.shape[0]): # lm.shape[0] = 68
            cv2.circle(img, (int(lm[i, 0]), int(lm[i, 1])), 1, (0, 255, 0), -1)
    # mouth_area
    mouth_left = lm[48, :]
    mouth_right = lm[54, :]
    mouth_x_mid = (mouth_left[0] + mouth_right[0]) // 2
    # 
    nose_top_mid = lm[29, :] # 鼻梁中心点
    if debug:
        print("nose_top_mid: ", nose_top_mid)
        print("lm[0:17, 1]: ", lm[0:17, 1])
    face_bottom = np.max(lm[0:17, 1], axis=0) # 下巴点
    # print("face_bottom: ", face_bottom)
    radius = max((face_bottom - nose_top_mid[1]), (mouth_right[0] - mouth_left[0])) * (1.1) // 2  # random.random() / 10 + 1.05
    radius = max(radius, 0)
    # print("radius: ", radius)
    if True:
        y1, y2 = nose_top_mid[1], nose_top_mid[1] + 2*radius + 1/4 * radius
        x1, x2 = mouth_x_mid - radius - 1/4 * radius, mouth_x_mid + radius + 1/4 * radius
    else:
        y1, y2 = nose_top_mid[1], nose_top_mid[1] + 2*radius
        x1, x2 = mouth_x_mid - radius, mouth_x_mid + radius
    if debug:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.imwrite("%s_face_lm_mouth.jpg" %(prefix), img)
        cv2.imwrite("%s_crop_mouths.jpg" %(prefix), img[int(y1):int(y2), int(x1):int(x2)])


    img_tensor = img_tensor.unsqueeze(0)
    mouth_area = img_tensor[:,:, int(y1):int(y2), int(x1):int(x2)]
    if mouth_area.shape[-1] > 0 and mouth_area.shape[-2] > 0:
        img_tensor = F.interpolate(mouth_area, size=(mouth_size, mouth_size), mode='bilinear', align_corners=True)
        msg = 0
    else:
        print("lm error ", img_tensor.shape, mouth_area.shape, "radius: ", radius, x1, y1, x2, y2, lm)
        torch_resize = Resize([mouth_size,mouth_size])
        img_tensor = torch_resize(img_tensor)
        msg = -1
    if debug:
        print("img_tensor: ", img_tensor.shape)
    return img_tensor.squeeze(0), msg, (int(y1),int(y2),int(x1),int(x2))


def save_sample_images(x_in, gt_im, mouth_y1y2x1x2, out_im, global_step, val_train="val", full_save=True):

    if len(gt_im) < 1:
        print("save_sample_images do nothing, return.")
        return

    clip_len = len(out_im)
    batch_size = out_im[0].size(0)

    if full_save:
        x_in = [torch.split(x, 1, dim=0) for x in x_in]
    gt_im = [torch.split(x, 1, dim=0) for x in gt_im]
    out_im = [torch.split(x, 1, dim=0) for x in out_im]
    # 
    clip_in_out_gt = []
    for i in range(batch_size):
        sample_in_list = []
        sample_out_list = []
        sample_gt_list = []
        for j in range(clip_len): # 5
            if full_save:
                sample_in_list.append(x_in[j][i])
            sample_out_list.append(out_im[j][i])
            sample_gt_list.append(gt_im[j][i])
        if full_save:
            sample_in_list = torch.cat(sample_in_list, dim=3).squeeze(0)
        sample_out_list = torch.cat(sample_out_list, dim=3).squeeze(0)
        sample_gt_list = torch.cat(sample_gt_list, dim=3).squeeze(0)
       
        if full_save:
            sample_in_out_gt = torch.cat([sample_in_list, sample_out_list, sample_gt_list], dim=2)
        else:
            sample_in_out_gt = torch.cat([sample_out_list, sample_gt_list], dim=2)
        sample_in_out_gt = (sample_in_out_gt.detach().cpu().numpy().transpose(1,2,0)* 255.).astype(np.uint8)
        clip_in_out_gt.append(sample_in_out_gt)
    clip_in_out_gt = np.concatenate(clip_in_out_gt, axis=0) # 2304 x 3840 x 3, 2304 = 256*9
 
    if val_train == "val":
        prefix = os.path.join("./temp/", "samples_step_val_{:06d}".format(global_step))
    else:
        prefix = os.path.join("./temp/", "samples_step_train_{:06d}".format(global_step))

    h, w = clip_in_out_gt.shape[:2]
    clip_in_out_gt = np.ascontiguousarray(clip_in_out_gt)
    pt1 = (w//3, 0)
    pt2 = (w//3, h)
    clip_in_out_gt = cv2.line(clip_in_out_gt, pt1, pt2, (0,0,255), 2)
    pt1 = (2*w//3, 0)
    pt2 = (2*w//3, h)
    clip_in_out_gt = cv2.line(clip_in_out_gt, pt1, pt2, (0,0,255), 2)
    cv2.imwrite('{}_{}.jpg'.format(prefix, global_step), clip_in_out_gt)
    print("saved sample_images: ", '{}_{}.jpg'.format(prefix, global_step))


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new




def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred.float(), dim=-1)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)
    # print("pitch: ", pitch.shape) # 1
    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    # print("pitch_mat: ", pitch_mat.shape) # 1, 9
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


# 输入 RGB float
def extract_face_bbox(frame, increase=0.1):
    ori_frame = frame
    import face_alignment

    face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        frame = img_as_ubyte(frame)
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = face_align.face_detector.detect_from_image(frame) #[..., ::-1]
    if len(bboxes) == 0:
        print("No face detected in frame")
        return []
    
    clx, cly, crx, cry = (np.array(bboxes)[:, :-1] * scale_factor).tolist()[0]
    
    frame_shape = ori_frame.shape[:2]
    box_w = crx - clx
    box_h = cry - cly
    width_increase = max(increase, ((1 + 2 * increase) * box_h - box_w) / (2 * box_w))
    height_increase = max(increase, ((1 + 2 * increase) * box_w - box_h) / (2 * box_h))
    clx = int(clx - width_increase * box_w)
    cly = int(cly - height_increase * box_h)
    crx = int(crx + width_increase * box_w)
    cry = int(cry + height_increase * box_h)
    cly, cry, clx, crx = max(0, cly), min(cry, frame_shape[0]), max(0, clx), min(crx, frame_shape[1])

    return clx, cly, crx, cry

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_model = models.vgg19(pretrained=True)
        vgg_pretrained_features = vgg_model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
