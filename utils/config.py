class Config:
    def __init__(self):
        # Directory Definitions
        self.CONTEXT_PATHS = {'train': 'train/', 'test': 'test/'}
        self.RAW_SOUND_DATA_PATHS = {'keyword': 'data/wallace/', 'background': 'data/background_noise/', 'non_keyword': 'data/other/'}
        self.GENERATED_DATA_PATHS = {'sound': 'data/generated/soundfiles/', 'label': 'data/generated/labels/'}
        self.MODEL_PATH = 'models/'

        # Sound Params
        self.WAV_FRAMERATE_HZ = 44100
        self.CLIP_LEN_MS = 10000
        self.RECORDING_CHUNK = 1024

        # Training Params
        self.TRAINING_SPLIT_PERCENT = 70
        self.BATCHES = 10
        self.BATCH_SIZE = 500

        # Pi Settings
        self.BUTTON_ID = 17
        self.RESPEAKER_CHANNELS = 1  # Mono/Stereo
        self.RESPEAKER_WIDTH = 2
        self.RESPEAKER_INDEX = 1  # input card ID
        self.RESPEAKER_FORMAT = 'int16'

        # Mac Settings
        self.MAC_CHANNELS = 1
        self.MAC_WIDTH = 2
        self.MAC_INDEX = 0
        self.MAC_FORMAT = 'int16'

        # Listener Settings
        self.MAX_RECORDING_FRAMES = 15
        self.MAX_SILENT_FRAMES = 5
