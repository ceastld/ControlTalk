import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from render import Renderer
from hparams import hparams as hp
from torchvision import models

# debug = True
debug = False

class ProjectionHead(nn.Module):
    def __init__(self,embedding_dim,projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x



class AudioEncoder(nn.Module):
    def __init__(self, freeze=True):
        super(AudioEncoder, self).__init__()
        self.fc1 = ProjectionHead(1024*hp.syncnet_audio_T*2,2048) # 10
        self.fc2 = ProjectionHead(2048,1024)
        self.fc3 = ProjectionHead(1024,512)
        self.fc4 = ProjectionHead(512,1024)
   
        if freeze:
            print("freeze audio encoder")
            for param in self.parameters():
                param.requires_grad = False
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # print(x.size())
        return x

class lipControlNet(nn.Module):
    def __init__(self, pretrained_syncnet_path, freeze_audio=True):
        super(lipControlNet, self).__init__()
        print("freeze_audio: ", freeze_audio)
        self.audio_encoder = AudioEncoder(freeze=freeze_audio)
        if freeze_audio:
            self.audio_encoder.eval()
        syncNet_state_dict = torch.load(pretrained_syncnet_path, map_location=torch.device('cpu'))
        state_dict = self.audio_encoder.state_dict()
        for k,v in syncNet_state_dict.items():
            if 'audio_encoder' in k:
                state_dict[k.replace("module.", "").replace('model.audio_encoder.', '')] = v
       
        self.audio_encoder.load_state_dict(state_dict, strict=True)
        print("loaded pretrained_syncnet_path: ", pretrained_syncnet_path)

 
        self.out_fc = nn.Linear(1024+3*15, 3*15)
        self.out_fc.weight.data.zero_()
        self.out_fc.bias.data.zero_()

    def forward(self, src_exp_in, drv_audio_fea, alpha=1.0):

        src_exp = src_exp_in.permute(0, 2, 1)
        try:
            src_exp = src_exp.view(src_exp.shape[0], -1)
        except:
            src_exp = src_exp.contiguous().view(src_exp.shape[0], -1)

        drv_audio_fea = drv_audio_fea.reshape(drv_audio_fea.shape[0], -1)
        drv_audio_fea = self.audio_encoder(drv_audio_fea.float())
        drv_audio_fea = drv_audio_fea.view(drv_audio_fea.size(0), -1)
        

        concat_fea = torch.concat([src_exp, drv_audio_fea], dim=1)
        delta_exp = self.out_fc(concat_fea)
        delta_exp = delta_exp.view(delta_exp.shape[0], -1, 3)
        out_exp = src_exp_in + delta_exp * alpha
        return out_exp

class lipControlFullModel(nn.Module):
    def __init__(self, lipCtrlNet, renderer):
        super(lipControlFullModel, self).__init__()
        self.lipCtrlNet = lipCtrlNet
        if renderer is not None:
            self.render = renderer
            for param in self.render.parameters():
                param.requires_grad = False
            self.render.eval()

    def forward(self, src_img_list, src_kp_list, src_he_list, src_indiv_hubert_list, gt_clip_he=None, gt_img_list=[], alpha=1.0, src_indiv_silence_hubert_list=None): # gt_clip_he  gt_img_list for_debug
        batch_size = src_indiv_hubert_list[0].shape[0]
        syncT = len(src_img_list)

        if True:
            hubert_fea = torch.cat(src_indiv_hubert_list, 0).squeeze(1)
            if src_indiv_silence_hubert_list is not None:
                silence_hubert_fea = torch.cat(src_indiv_silence_hubert_list, 0).squeeze(1)
            src_exp = torch.cat([x["exp"] for x in src_he_list], 0).squeeze(1)
            if np.random.rand() < 0.5 and src_indiv_silence_hubert_list is not None:
                exp_out = self.lipCtrlNet(src_exp, silence_hubert_fea, alpha)
                exp_out = self.lipCtrlNet(exp_out, hubert_fea, alpha)
            else:
                exp_out = self.lipCtrlNet(src_exp, hubert_fea, alpha)
    
            drv_exp_list = torch.split(exp_out, batch_size, dim=0)

            if debug:
                print("gt_clip_he: ", len(gt_clip_he), gt_clip_he[0]["exp"].shape, gt_clip_he[0]["exp"].device)
                drv_exp_list = [x["exp"] for x in gt_clip_he] # for_debug

            if len(src_img_list) == 0:
                return [], drv_exp_list

            drv_he_list = deepcopy(src_he_list)
            outputs = []
            for i in range(syncT):
                drv_he_list[i]["exp"] = drv_exp_list[i]
                src_he = src_he_list[i]
                drv_he = drv_he_list[i]
         
                out = self.render.forward_kp_he(src_img_list[i][:, (2, 1, 0)], src_kp_list[i], src_he, drv_he)
                outputs.append(out['prediction'][:, (2, 1, 0)])

            else:
                return outputs, drv_exp_list

