class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters
hparams = HParams(
    wav2lip_audio_T=9,
    syncnet_audio_T=9,
    example_wav = 'data/example.wav',
    audio_encoder_path = 'checkpoints/audio_encoder.pt',
    hubert_checkpoint_path = 'checkpoints/TencentGameMate/chinese-hubert-large',
    lipControlNet_checkpoint_path = './checkpoints/lipControlNet.pt',
    render_config = "./configs/vox-256-beta.yaml",
    render_ckpt = "./checkpoints/20231128_210236_337a_e0362-checkpoint.pth.tar",

)


def hparams_debug_string():
    print("Hyperparameters batch_size:", hparams.batch_size)
    values = hparams.data
    # print("  values: %s" % values)
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)


if __name__ == '__main__':
    print(hparams_debug_string())
    exit(0)

