import numpy as np
from pydub import AudioSegment
import random
import os
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as pyplt
import utils.config as conf
from datetime import datetime

class TrainUtil:
    def __init__(self):
        # import config
        config = conf.Config()
        self.context_paths = config.CONTEXT_PATHS
        self.import_data_paths = config.GENERATED_DATA_PATHS
        self.model_path = config.MODEL_PATH
        self.framerate = config.WAV_FRAMERATE_HZ
        self.ty = config.TY
        self.clip_len_ms = config.CLIP_LEN_MS
        self.training_split = config.TRAINING_SPLIT_PERCENT
        self.seed = config.SEED


        # set seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Fire
        self.first()

    def first(self):
        # load train data
        spectrograms_train, labels_train = self.load_data('train')
        print(spectrograms_train.shape)
        print(labels_train.shape)

        model = self.train(spectrograms_train, labels_train)
        spectrograms_test, labels_test = self.load_data('test')
        model.evaluate(spectrograms_test, labels_test)

        self.save_model(model)

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

    def convert_label(self, label):
        converted_label = np.zeros((1, self.ty))
        for sequence in label:
            # add label at end time
            converted_label = self.add_label(converted_label, sequence[1])

        return converted_label

    def add_label(self, y, segment_end_ms):
        segment_end_y = int(segment_end_ms * self.ty / self.clip_len_ms)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < self.ty:
                y[0, i] = 1
        return y


    def train(self, spectrograms, labels):
        model = models.Sequential([
            layers.Input(shape=(5511, 101)),
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
        model.fit(spectrograms, labels, epochs=10)
        return model

    def load_data(self, context):
        path = self.import_data_paths['sound']+self.context_paths[context]
        sound = []
        imported_sounds = []
        for filename in os.listdir(path):
            if filename.endswith("wav"):
                sound.append(self.get_spectrogram(path + filename))
                imported_sounds.append(os.path.splitext(filename)[0])

        path = self.import_data_paths['label']+self.context_paths[context]
        label = []
        for filename in imported_sounds:
            label.append(self.convert_label(np.load(path + filename + '.npy')))

        sound = np.swapaxes(np.array(sound), 0, 1)
        label = np.swapaxes(np.array(label), 0, 1)
        return sound, label

    def save_model(self, model):
        model.save(self.model_path+'wallace_activation_'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+'.h5')

if __name__ == '__main__':
    TrainUtil()
