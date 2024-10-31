import os
import time
import pickle
import cv2
import argparse
import torch
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from render import Renderer
from model import lipControlNet
from hparams import hparams as hp
from local_utils import extract_face_bbox

torch.set_printoptions(precision=10)

parser = argparse.ArgumentParser(description='Inference code to generate video by control talk')

parser.add_argument('--checkpoint_path', type=str, default=hp.lipControlNet_checkpoint_path,
                    help='Name of saved checkpoint to load weights from', required=False)
parser.add_argument('--source_img_path', type=str,
                    help='Filepath of image to use for driving source face', required=False)
parser.add_argument('--source_video', type=str, 
                    help='Filepath of video that contains faces to use', required=False)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--silence_audio', type=str, default=hp.example_wav,
                    help='Filepath of video/audio file to use as raw audio source', required=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)
parser.add_argument('--alpha', type=float, help='Can control the lip movement', 
                    default=0.7, required=False)
parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=16)
parser.add_argument('--infer_batch_size', type=int, help='Batch size for lipControlFullModel model(s)', default=4)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')
parser.add_argument('--img_mode', default=False, action='store_true',
                    help='Use when feeding an image instead of a video')
parser.add_argument('--save_as_video', action="store_true", default=False,
                    help='Whether to save frames as video', required=False)
parser.add_argument('--concat_out', action="store_true", default=False,
                    help='Whether to concat input and output images', required=False)
parser.add_argument('--img_size', type=int, default=256, required=False,
                    help='Size of image to use (default: 256)')
parser.add_argument('--render_config', type=str, default=hp.render_config,
                    help='Path to config for loading model')
parser.add_argument('--render_ckpt', type=str, default=hp.render_ckpt,
                    help='Path to checkpoint for loading model')
parser.add_argument('--use_amp', help='Freeze audio encoder weights', action='store_true')

args = parser.parse_args()

def face_detect(images):
    first_frame = images[0]
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    box = extract_face_bbox(first_frame, increase=0.2)
    return box 

def datagen(frames, face_det_results, clip_features, silence_feature, use_silence=False):
    face_batch, fea_batch, frame_batch, coords_batch = [], [], [], []
    silence_fea_batch = []

    for i, fea in enumerate(clip_features):
        
        if (i // len(frames)) % 2 == 0:
            idx = i%len(frames) 
        else:
            idx = len(frames) - i%len(frames) - 1  

        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size), interpolation = cv2.INTER_CUBIC)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # 

        face_batch.append(face)
        fea_batch.append(fea)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if use_silence:
            silence_fea_batch.append(silence_feature)

        if len(face_batch) >= args.infer_batch_size:
            face_batch = np.asarray(face_batch)
            face_batch = face_batch / 255.
            if args.use_amp:
                face_batch = torch.HalfTensor(np.transpose(face_batch, (0, 3, 1, 2)))
                fea_batch = torch.cat([torch.HalfTensor(f).unsqueeze(0) for f in fea_batch], dim=0)
                silence_fea_batch= torch.cat([torch.HalfTensor(f).unsqueeze(0) for f in silence_fea_batch], dim=0)
            else:
                face_batch = torch.FloatTensor(np.transpose(face_batch, (0, 3, 1, 2)))
                fea_batch = torch.cat([torch.FloatTensor(f).unsqueeze(0) for f in fea_batch], dim=0)
                silence_fea_batch= torch.cat([torch.FloatTensor(f).unsqueeze(0) for f in silence_fea_batch], dim=0)

            yield face_batch, fea_batch, frame_batch, coords_batch, silence_fea_batch
            face_batch, fea_batch, frame_batch, coords_batch = [], [], [], []
            silence_fea_batch = []

    if len(face_batch) > 0:
        print("face_batch: ", len(face_batch), "fea_batch: ", len(fea_batch))
        face_batch, fea_batch = np.asarray(face_batch), np.asarray(fea_batch)
        silence_fea_batch = np.asarray(silence_fea_batch)
        face_batch = face_batch / 255.
        if args.use_amp:
            face_batch = torch.HalfTensor(np.transpose(face_batch, (0, 3, 1, 2)))
            fea_batch = torch.cat([torch.HalfTensor(f).unsqueeze(0) for f in fea_batch], dim=0)
            silence_fea_batch= torch.cat([torch.HalfTensor(f).unsqueeze(0) for f in silence_fea_batch], dim=0)
        else:
            face_batch = torch.FloatTensor(np.transpose(face_batch, (0, 3, 1, 2)))
            fea_batch = torch.cat([torch.FloatTensor(f).unsqueeze(0) for f in fea_batch], dim=0)
            silence_fea_batch= torch.cat([torch.FloatTensor(f).unsqueeze(0) for f in silence_fea_batch], dim=0)

        yield face_batch, fea_batch, frame_batch, coords_batch, silence_fea_batch

fea_step_size = 2*hp.wav2lip_audio_T # 18 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
  
    return checkpoint

def load_model(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    if "state_dict" in checkpoint:
        s = checkpoint["state_dict"]
    else:
        s = checkpoint
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

if __name__ == '__main__':
    if not os.path.exists("temp"):
        os.mkdir("temp")
    else:
        os.system("rm -rf temp/*")
    if not os.path.exists("results"):
        os.mkdir("results")

    lipCtrlnet = lipControlNet(hp.audio_encoder_path)
    lipCtrlnet.cuda()
    lipCtrlnet.eval()
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        load_model(args.checkpoint_path, lipCtrlnet)
        print("load model_checkpoint: ", args.checkpoint_path)
    else:
        print("args.checkpoint_path error!")
        exit()
    renderer = Renderer(config_path=args.render_config, 
                               checkpoint_path=args.render_ckpt, 
                               pic_size=args.img_size,
                               gen="spade")
    renderer.cuda()
    renderer.eval()
    print ("Model loaded")

    # 
    src_audio = args.audio
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        os.system(command)
        args.audio = 'temp/temp.wav'
    
    # 
    audio_fea_path = os.path.join("temp", src_audio.rsplit('/')[-1].rsplit('.')[0] + "_audio_feature.npy")
    silence_audio_fea_path = os.path.join("temp", args.silence_audio.rsplit('/')[-1].rsplit('.')[0] + "_audio_feature.npy")

    if not os.path.exists(silence_audio_fea_path):
        from modules.tencent_wav2vec import extract_feature_tx_hubert
        feature_silence = extract_feature_tx_hubert(args.silence_audio, set_zero=False)
        feature_silence = feature_silence.T
        np.save(silence_audio_fea_path, feature_silence)
    else:
        feature_silence = np.load(silence_audio_fea_path)

    if not os.path.exists(audio_fea_path):
        from modules.tencent_wav2vec import extract_feature_tx_hubert
        print("Extracting feature from tencent hubert...")
        feature = extract_feature_tx_hubert(args.audio)
        feature = feature.T
        np.save(audio_fea_path, feature)
    else:
        feature = np.load(audio_fea_path)
    print("audio feature: ", feature.shape, len(feature[0])) 
    fea_chunks = []
    fea_silence_chunks = []
    fps = args.fps
    mel_idx_multiplier = 50./fps  
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + fea_step_size > len(feature[0]):
            fea_chunks.append(feature[:, len(feature[0]) - fea_step_size:].T)
            fea_silence_chunks.append(feature_silence[:, :fea_step_size].T)
            break
        fea_chunks.append(feature[:, start_idx : start_idx + fea_step_size].T)
        fea_silence_chunks.append(feature_silence[:, :fea_step_size].T)
        i += 1
    print("Length of feature chunks: {}".format(len(fea_chunks)), fea_chunks[0].shape) # 247

    # 
    if os.path.isdir(args.source_video):
        import glob
        full_frames = []
        for imfile in sorted(glob.glob(os.path.join(args.source_video, '*.png'))):
            frame = cv2.imread(imfile)
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor), interpolation = cv2.INTER_CUBIC)
            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    elif os.path.isfile(args.source_video):
        video_stream = cv2.VideoCapture(args.source_video)
        h = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        print('===== Video size: {}x{}'.format(w, h))
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        if 25 != fps:
            print("!!! fps: ", fps)
        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor), interpolation = cv2.INTER_CUBIC)
            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    else:
        print("args.source_video error!")
        exit()
    print ("Number of frames available for inference: "+str(len(full_frames)), full_frames[0].shape)

    if args.img_mode:
        full_frames = [full_frames[0]]*len(fea_chunks)
    else:
        full_frames = full_frames[:len(fea_chunks)] 

    if len(full_frames) < len(fea_chunks):
        full_frames_cycle = full_frames + full_frames[::-1]
        full_frames = full_frames_cycle * (len(fea_chunks) // len(full_frames_cycle)) + full_frames_cycle[:len(fea_chunks) % len(full_frames_cycle)]

    if args.box[0] == -1:  
        tface0 = time.time()
        save_face_box_path = os.path.join("temp", args.source_video.rsplit('/')[-1].rsplit('.')[0] + "_face_bbox.pickle")
        if os.path.exists(save_face_box_path):
            print("load face boxes from: ", save_face_box_path)
            box = pickle.load(open(save_face_box_path, "rb"))
        else:
            box = face_detect(full_frames) # BGR2RGB for CNN face detection
            pickle.dump(box, open(save_face_box_path, "wb"))
        print("box: ", box) # clx, cly, crx, cry
        boxes = [box] * len(full_frames)
        print("face_detect time cost: ", time.time() - tface0)
    else:
        boxes = [args.box] * len(full_frames)
    print(full_frames[0].shape)
    face_det_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(full_frames, boxes)]
    print("face_det_results: ", len(face_det_results), face_det_results[0][0].shape)

    source_face = None
    if args.source_img_path:
        source_img = cv2.imread(args.source_img_path)
        x1,y1,x2,y2 = face_detect([source_img])
        print("source_img: ", source_img.shape, x1,y1,x2,y2)
        source_face = source_img[y1:y2, x1:x2]
        source_face = cv2.resize(source_face, (args.img_size, args.img_size))
        source_face = cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB)
        source_face_batch = np.asarray([source_face] * args.infer_batch_size)
        source_face_batch = source_face_batch / 255.
        source_face_batch = torch.FloatTensor(np.transpose(source_face_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            source_face_he, source_face_kp = renderer(None, source_face_batch, extract_driving_only=True)
        print("source_face_he: ", source_face_he["exp"].shape, source_face_kp["value"].shape)

    batch_size = args.infer_batch_size
    batch_gen = datagen(full_frames.copy(), face_det_results, fea_chunks, fea_silence_chunks[0], use_silence=True)
 
    # 
    if args.save_as_video:
        if args.concat_out:
            pred_out = cv2.VideoWriter("temp/pred.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (args.img_size*2, args.img_size))
        else:
            h, w = full_frames[0].shape[:2]
            pred_out = cv2.VideoWriter("temp/pred.avi", cv2.VideoWriter_fourcc(*'DIVX'), 25, (args.img_size, args.img_size))
    t0 = time.time()
    count = 0
    for i, (face_batch, fea_batch, frames, coords, fea_silence_batch) in enumerate(tqdm(batch_gen, "infering...",
                                            total=int(np.ceil(float(len(fea_chunks))/batch_size)))):
        face_batch = face_batch.to(device)
        fea_batch = fea_batch.to(device) 
        fea_silence_batch = fea_silence_batch.to(device)  
        with torch.no_grad():
            src_he, src_kp = renderer(None, face_batch, extract_driving_only=True)
            src_he["exp"] = src_he["exp"].view(face_batch.shape[0], -1, 3)
            drv_he = deepcopy(src_he)
            src_exp = src_he["exp"].clone()
            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    exp_out = lipCtrlnet(src_exp, fea_silence_batch, alpha=1.0)
                    exp_out = lipCtrlnet(exp_out, fea_batch, alpha=args.alpha)
            else:
                exp_out = lipCtrlnet(src_exp, fea_silence_batch, alpha=1.0)
                exp_out = lipCtrlnet(exp_out, fea_batch, alpha=args.alpha)
            drv_he["exp"] = exp_out
            if source_face is not None:
                src_kp = {k:v[:len(src_kp[k])] for k,v in source_face_kp.items()}
                src_he = {k:v[:len(src_he[k])] for k,v in source_face_he.items()}
                infer_batch = source_face_batch[:len(face_batch)]
            else:
                infer_batch = face_batch
            pred = renderer.forward_kp_he(infer_batch, src_kp, src_he, drv_he)["prediction"]
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for j, p in enumerate(pred):
            if args.save_as_video:
                if args.concat_out: 
                    in_img = face_batch[j].cpu().numpy().transpose(1, 2, 0) * 255.
                    in_img = in_img.astype(np.uint8)[:, :, ::-1]
                    out_img = p.astype(np.uint8)[:, :, ::-1]
                    concat_img = np.concatenate([in_img, out_img], axis=1)
                    pred_out.write(concat_img)
                else:
                    pred_out.write(p.astype(np.uint8)[:, :, ::-1])
                count += 1

    t1 = time.time()
    print("count: ", count)    
    print("Total time taken: {}".format(t1-t0))    
    print("i, len(fea_chunks): ", i, len(fea_chunks)) 
    print("Avg time per frame: {}".format((t1-t0)/float(len(fea_chunks)))) 

    outfile = os.path.join("results", args.audio.rsplit('/')[-1].rsplit('.')[0] + "-" \
                            + args.source_video.rsplit('/')[-1].rsplit('.')[0]  + "-" \
                            + args.checkpoint_path.rsplit('/')[-1].rsplit('.',1)[0] + "-" \
                            + str(args.img_mode) + "_" + str(args.img_size) \
                            + ".mp4")
    if os.path.exists("temp/pred.avi"):
        command = 'ffmpeg -i {} -i {} -q:v 2 {} -y'.format('temp/pred.avi', args.audio, outfile)
        print("=== command ===: \n", command)
        os.system(command)
    print("finish!")


