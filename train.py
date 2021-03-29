import numpy as np
import os
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as pyplt
import utils.config as conf
from datetime import datetime
import json

class Train:
    def __init__(self):
        # import config
        config = conf.Config()
        self.context_paths = config.CONTEXT_PATHS
        self.import_data_paths = config.GENERATED_DATA_PATHS
        self.model_path = config.MODEL_PATH
        self.framerate = config.WAV_FRAMERATE_HZ
        self.clip_len_ms = config.CLIP_LEN_MS
        self.training_split = config.TRAINING_SPLIT_PERCENT
        self.seed = config.SEED
        self.batch_size = 8
        self.epochs = 100

        self.sound_shape = None
        self.model = None

    def run(self):
        print('get train and test files')

        train_files = self.get_sound_filenames('train')
        test_files = self.get_sound_filenames('test')


        print('get train and test loaders')
        train_loader = Loader(train_files, self.get_label_shape(self.get_model()), self.batch_size, 'train')
        test_loader = Loader(test_files, self.get_label_shape(self.get_model()), self.batch_size, 'test')

        model, history = self.train(train_loader, test_loader)

        print('get eval files and loader')
        eval_files = self.get_sound_filenames('eval')
        test_loader = Loader(eval_files, self.get_label_shape(self.get_model()), self.batch_size, 'eval')
        model.evaluate_generator(test_loader)

        self.save_model(model, history)

    def train(self, train_loader, test_loader):
        model = self.get_model()
        history = model.fit_generator(generator=train_loader, epochs=self.epochs, use_multiprocessing=False, verbose=1, validation_data=test_loader)
        return model, history

    def get_model(self):
        if self.model is None:
            model = models.Sequential([
                layers.Input(shape=self.get_sound_shape()),
                layers.Conv1D(196, kernel_size=15, strides=4, activation=tf.nn.relu),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dropout(0.8),
                layers.GRU(units=128, return_sequences=True),
                layers.Dropout(0.8),
                layers.BatchNormalization(),
                layers.GRU(units=128, return_sequences=True),
                layers.Dropout(0.8),
                layers.BatchNormalization(),
                layers.Dropout(0.8),
                layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
                ])
            opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            model.summary()
            self.model = model
        return self.model

    def get_spectrogram(self, sound):
        rate, data = wavfile.read(sound)
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = pyplt.specgram(data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = pyplt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx

    def get_sound_filenames(self, context):
        path = self.import_data_paths['sound'] + self.context_paths[context]
        sound_paths = []
        for filename in os.listdir(path):
            if filename.endswith("wav"):
                # only add filename without extension
                sound_paths.append(os.path.splitext(filename)[0])
        return sound_paths

    def get_sound_shape(self):
        # TODO: Refactor me, I'm ugly..
        if self.sound_shape is None:
            # considering there is at least one training file present. The shape represents all datasets, since generated..
            sound = np.swapaxes(np.array(self.get_spectrogram(self.import_data_paths['sound'] + self.context_paths['train'] + '1.wav')), 0, 1)
            self.sound_shape = sound.shape
        return self.sound_shape

    def get_label_shape(self, model):
        layers = model.layers
        label_shape = layers[0].output_shape
        return label_shape[1]

    def save_model(self, model, history):
        model.save(self.model_path+'wallace_activation_'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+'.h5')
        with open(self.model_path+'wallace_activation_'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+'.json', 'w') as historyfile:
            json.dump(history.history, historyfile)

class Loader(tf.keras.utils.Sequence):
    def __init__(self, soundfiles, label_shape, batch_size, context):
        config = conf.Config()
        self.soundfiles = soundfiles
        self.label_shape = label_shape
        self.batch_size = batch_size
        self.context = context
        self.import_data_paths = config.GENERATED_DATA_PATHS
        self.context_paths = config.CONTEXT_PATHS
        self.clip_len_ms = config.CLIP_LEN_MS

    def __len__(self):
        return (np.ceil(len(self.soundfiles) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, item):
        batch = self.soundfiles[item * self.batch_size : (item + 1) * self.batch_size]
        sound_batch = []
        label_batch = []

        for filename in batch:
            sound, label = self.load_data(filename)
            sound_batch.append(sound)
            label_batch.append(label)

        return np.array(sound_batch), np.array(label_batch)

    def load_data(self, filename):
        sound_path = self.import_data_paths['sound'] + self.context_paths[self.context]
        label_path = self.import_data_paths['label'] + self.context_paths[self.context]
        sound = np.swapaxes(np.array(self.get_spectrogram(sound_path + filename + '.wav')), 0, 1)
        label = np.swapaxes(np.array(self.convert_label(self.label_shape, np.load(label_path + filename + '.npy'))), 0, 1)

        return sound, label

    def get_spectrogram(self, sound):
        rate, data = wavfile.read(sound)
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = pyplt.specgram(data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = pyplt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx

    def convert_label(self, label_shape, label):
        converted_label = np.zeros((1, label_shape))
        for sequence in label:
            # add label at end time
            converted_label = self.add_label(converted_label, sequence[1])

        return converted_label

    def add_label(self, y, segment_end_ms):
        label_shape = y.shape[1]
        segment_end_y = int(segment_end_ms * label_shape / self.clip_len_ms)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < label_shape:
                y[0, i] = 1
        return y

if __name__ == '__main__':
    tm = Train()
    tm.run()
