import torch
import numpy as np
import librosa
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from hparams import hparams as hp
processor = Wav2Vec2FeatureExtractor.from_pretrained(hp.hubert_checkpoint_path)
model = HubertModel.from_pretrained(hp.hubert_checkpoint_path).cuda()

def extract_feature_tx_hubert(audio_path, set_zero=False):
    sample_array = librosa.load(audio_path,sr=16_000)[0]
    if set_zero:
        sample_array = np.zeros_like(sample_array)
    inputs = processor(sample_array, sampling_rate=16000, return_tensors="pt")
    inputs['input_values'] = inputs['input_values'].cuda()
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.cpu().numpy()
