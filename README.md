# ControlTalk
[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/NetEase-Media/ControlTalk)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2406.02880)

Official code for "Controllable Talking Face Generation by Implicit Facial Keypoints Editing"


**ControlTalk**: A talking face generation method to control face expression deformation based on driven
audio, constructing the head pose and facial expression (lip motion) for both single image or sequential video inputs in a unified manner.

<video src="https://github.com/user-attachments/assets/7c01c7ed-e6c6-482c-a39b-86072428ea9c">  </video>

## News
- **2024/10/31**: Inference code is now available!

## Installation

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda env create -f environment.yml
```

## Inference

### 1. Download checkpoints
- Download pretrained models from huggingface [detailed guidance](https://huggingface.co/docs/huggingface_hub/guides/download).

```bash
# Download hubert model
https://huggingface.co/TencentGameMate/chinese-hubert-large

# Download our pretrained model 
https://huggingface.co/Lavivis/ControlTalk
```
- Put all pretrained models in `./checkpoints`, the file structure should be like:
```
checkpoints
├── audio_encoder.pt
├── lipControlNet.pt
├── 20231128_210236_337a_e0362-checkpoint.pth.tar
├── TencentGameMate
├───└──chinese-hubert-large
├─────────└──config.json
├─────────└──pytorch_model.bin
├─────────└──preprocessor_config.json
└─────────└──chinese-hubert-large-fairseq-ckpt.pt
```
### 2. Inference

```bash
python inference.py \
        --source_video './data/drive_video.mp4' \
        --source_img_path  './data/example.png' \
        --audio './data/drive_audio.wav' \
        --save_as_video \
        --box -1 0 0 0 \
        # --img_mode   # if you only want to control the face expression
```


## Training

Coming soon!


## Acknowledgements
- [chinese_speech_pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [face-vid2vid](https://nvlabs.github.io/face-vid2vid/)
- [face-vid2vid (Unofficial implementation)](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)


## Citation
If our work and codebase is useful for you, please cite as:
```
@article{zhao2024controllable,
  title={Controllable Talking Face Generation by Implicit Facial Keypoints Editing},
  author={Zhao, Dong and Shi, Jiaying and Li, Wenjun and Wang, Shudong and Xu, Shenghui and Pan, Zhaoming},
  journal={arXiv preprint arXiv:2406.02880},
  year={2024}
}
```
## License

Our code is released under MIT License. 