ml CUDA/12.4.0
python inference.py \
        --source_video './data/drive_video.mp4' \
        --source_img_path  './data/example.png' \
        --audio './data/drive_audio.wav' \
        --save_as_video \
        --box -1 0 0 0 \
        # --img_mode   # if you only want to control the face expression