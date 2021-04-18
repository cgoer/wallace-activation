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
    def __init__(self, batch, model_filename=None):
        # import config
        config = conf.Config()
        self.context_paths = config.CONTEXT_PATHS
        self.import_data_paths = config.GENERATED_DATA_PATHS
        self.model_path = config.MODEL_PATH
        self.framerate = config.WAV_FRAMERATE_HZ
        self.clip_len_ms = config.CLIP_LEN_MS
        self.training_split = config.TRAINING_SPLIT_PERCENT
        self.batch = batch
        self.model_filename = model_filename
        self.epochs = 50

        self.sound_shape = None
        self.model = None

    def run(self):
        # load train data
        print('load train data')
        spectrograms_train, labels_train = self.load_data('train', self.batch)
        print(spectrograms_train.shape)
        print(labels_train.shape)
        spectrograms_test, labels_test = self.load_data('test', self.batch)
        self.model, history = self.train(spectrograms_train, labels_train, spectrograms_test, labels_test)
        # wipe data
        spectrograms_train, labels_train, spectrograms_test, labels_test = None, None, None, None
        spectrograms_eval, labels_eval = self.load_data('eval', self.batch)
        print('evaluating model')
        self.model.evaluate(spectrograms_eval, labels_eval)
        # wipe data
        spectrograms_eval, labels_eval = None, None
        return self.save_model(self.model, history, self.batch)

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


    def train(self, spectrograms, labels, spectrograms_test, labels_test):
        model = self.get_model()
        history = model.fit(spectrograms, labels, epochs=self.epochs, validation_data=(spectrograms_test, labels_test))
        return model, history

    def get_model(self):
        # Return cached model
        if self.model is not None:
            return self.model

        # load certain model if provided
        if self.model_filename is not None:
            return self.load_model()

        # define new model
        if self.sound_shape is None:
            exit(1)
        return self.set_model(self.sound_shape)


    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_filename)
        return self.model

    def set_model(self, sound_shape):
        model = models.Sequential([
            layers.Input(shape=sound_shape),
            layers.Conv1D(196, kernel_size=15, activation=tf.nn.relu),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool1D(pool_size=2, strides=2, padding='valid'),
            layers.Conv1D(196, kernel_size=15, activation=tf.nn.relu),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool1D(pool_size=2, strides=2, padding='valid'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            layers.GRU(units=128, return_sequences=True),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.GRU(units=128, return_sequences=True),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
            ])
        opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model
        return self.model

    def load_data(self, context, batch_no):
        path = self.import_data_paths['sound']+self.context_paths[context]+str(batch_no)+'/'
        sounds = []
        imported_sounds = []

        for filename in os.listdir(path):
            if filename.endswith("wav"):
                sound = np.swapaxes(np.array(self.get_spectrogram(path + filename)), 0, 1)
                self.sound_shape = sound.shape
                sounds.append(sound)
                imported_sounds.append(os.path.splitext(filename)[0])

        model = self.get_model()
        layers = model.layers

        label_shape = layers[15].output_shape
        label_shape = label_shape[1]

        path = self.import_data_paths['label']+self.context_paths[context]+str(batch_no)+'/'
        labels = []
        for filename in imported_sounds:
            label = self.convert_label(label_shape, np.load(path + filename + '.npy'))
            labels.append(np.swapaxes(np.array(label), 0, 1))

        return np.array(sounds), np.array(labels)

    def save_model(self, model, history, batch):
        model_path = self.model_path+'wallace_activation_batch'+str(batch)+'_'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        model.save(model_path)

        # Save history
        with open(model_path+'.json', 'w') as historyfile:
            json.dump(history.history, historyfile)

        # Convert and save as tflite model
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)  # path to the SavedModel directory
        tflite_model = converter.convert()
        with open(model_path+'.tflite', 'wb') as f:
            f.write(tflite_model)

        return model_path

if __name__ == '__main__':
    config = conf.Config()
    batches = config.BATCHES
    model_path = None
    start_batch = 0
    for batch in range(batches):
        if batch < start_batch:
            continue
        print('Starting Batch '+str(batch)+'/'+str(batches))
        tm = Train(batch, model_path)
        model_path = tm.run()
