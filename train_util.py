import numpy as np
from pydub import AudioSegment
import random
import os
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as pyplt
import config as conf
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
        model = self.train(spectrograms_train, labels_train)
        spectrograms_test, labels_test = self.load_data('test')
        model.evaluate(spectrograms_test, labels_test)

        self.save_model(model)



    def train(self, spectrograms, labels):
        model = tf.keras.models.Sequential([
            #tf.keras.layers.Input(5511, 111),
            tf.keras.layers.Conv1D(196, kernel_size=15, strides=4, input_shape=(5511, 101), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.GRU(units=128, return_sequences=True),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GRU(units=128, return_sequences=True),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(1, activation='sigmoid')
            )
            ])
        opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(spectrograms, labels, batch_size=5, epochs=10)
        return model

    def load_data(self, context):
        spectrograms = np.load(self.import_data_paths['numpy'] + self.context_paths[context] + 'spectrograms.npy')
        labels = np.load(self.import_data_paths['numpy'] + self.context_paths[context] + 'labels.npy')
        return spectrograms, labels

    def save_model(self, model):
        tf.keras.models.save_model(model, self.model_path+'wallace_activation_'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

if __name__ == '__main__':
    TrainUtil()
