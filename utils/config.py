class Config:
    def __init__(self):
        # Directory Definitions
        self.CONTEXT_PATHS = {'train': 'train/', 'test': 'test/'}
        self.RAW_SOUND_DATA_PATHS = {'keyword': 'data/wallace/', 'background': 'data/background_noise/', 'non_keyword': 'data/other/'}
        self.GENERATED_DATA_PATHS = {'sound': 'data/generated/soundfiles/', 'numpy': 'data/generated/numpyfiles/'}
        self.MODEL_PATH = 'models/'

        # Sound Params
        self.WAV_FRAMERATE_HZ = 44100
        self.TY = 1375
        self.CLIP_LEN_MS = 10000

        # Training Params
        self.TRAINING_SPLIT_PERCENT = 70
        self.SEED = 666

        # Pi Settings
        self.BUTTON_ID = 17
        self.RESPEAKER_CHANNELS = 1  # Mono/Stereo
        self.RESPEAKER_WIDTH = 2
        self.RESPEAKER_INDEX = 1  # input card ID
