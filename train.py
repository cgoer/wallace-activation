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
    """
    Train Class uses the generated data from sample_generator and fts them onto a defined model.
    It will run one batch provided and return the filename of the generated model.
    """
    def __init__(self, batch, model_filename=None):
        """
        :param int batch: The current batch number to use for incoming data
        :param str|None model_filename: A Filepath to a model to re-train
        """
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
        self.epochs = 10

        self.sound_shape = None
        self.model = None

    def run(self):
        """
        :returns str: The Filepath of the generated model
        """
        # load train data
        print('load train data')
        spectrograms_train, labels_train = self.load_data('train', self.batch)
        print('Done. Loaded '+str(len(labels_train))+' items.')
        print('load test data')
        spectrograms_test, labels_test = self.load_data('test', self.batch)
        print('Done. Loaded ' + str(len(labels_test)) + ' items.')

        print('Start Training.')
        print('----------')
        self.model, history = self.train(spectrograms_train, labels_train, spectrograms_test, labels_test)

        print('done.')
        print('----------')
        return self.save_model(self.model, history, self.batch)

    @staticmethod
    def get_spectrogram(sound):
        """
        Loads a soundfile and returns spectrogram
        :param str sound: Filepath to wavfile
        :returns np.array: Spectrogram as np.array
        """
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
        """
        Converts a label from timestep in milliseconds to timesteps in a desired shape
        :param np.array.shape label_shape: desired shape
        :param np.array label: array of arrays containing start- and end-time
        """
        converted_label = np.zeros((1, label_shape))
        for sequence in label:
            # add label at end time
            converted_label = self.add_label(converted_label,sequence[0], sequence[1])

        return converted_label

    def add_label(self, label, segment_start_ms, segment_end_ms):
        """
        Helper function for convert_label doing the actual conversion
        :param np.array label: a np array representing the label in the final format
        :param int segment_start_ms: The start time in ms
        :param int segment_end_ms: The end time in ms
        """
        label_shape = label.shape[1]
        segment_start = int(segment_start_ms * label_shape / self.clip_len_ms)
        segment_end = int(segment_end_ms * label_shape / self.clip_len_ms)
        for i in range(segment_start, segment_end):
            if i < label_shape:
                label[0, i] = 1
        return label

    def train(self, spectrograms, labels, spectrograms_test, labels_test):
        """
        Fits Data to the model.
        :param np.array spectrograms: A numpy array of spectrograms (training data)
        :param np.array labels: A np.array of labels (traning data)
        :param np.array spectrograms_test: A numpy array of spectrograms (test data)
        :param np.array labels_test: A np.array of labels (test data)
        :returns: Model Object and History object of the fitting iteration
        """
        model = self.get_model()
        history = model.fit(spectrograms, labels, epochs=self.epochs, validation_data=(spectrograms_test, labels_test))
        return model, history

    def get_model(self):
        """
        Returns a model depending of current status of the training suite.
        :returns tf.keras.models: tf.keras.models object
        """
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
        """
        Loads a model from a given filepath
        :returns tf.keras.models: tf.keras.models
        """
        self.model = tf.keras.models.load_model(self.model_filename)
        return self.model

    def set_model(self, sound_shape):
        """
        Defines a net tf.keras.models model
        :params np.array sound_shape: The shape of inputs
        :returns tf.keras.models: Returns a tf.keras.models object
        """
        model = models.Sequential([
            layers.Input(shape=sound_shape),
            layers.Conv1D(196, kernel_size=15, activation=tf.nn.relu),
            layers.Conv1D(196, kernel_size=15, activation=tf.nn.relu),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.MaxPool1D(pool_size=2, strides=4, padding='valid'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.4),
            layers.GRU(units=128, return_sequences=True),
            layers.Dropout(0.4),
            layers.BatchNormalization(),
            layers.GRU(units=128, return_sequences=True),
            layers.Dropout(0.4),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
            ])
        opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model
        return self.model

    def load_data(self, context, batch_no):
        """
        Loads the data in a specific batch and context (e.g. Train/Test) from the desired directory.
        Also loads the model to get the current label shape.
        :params str context: The Context currently used (Train/test)
        :params int batch_no: Current batch number to use the correct directory
        :returns np.array: np.arrays of all sounds and labels
        """
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
        """
        Saves model and history, converts model to tflite and also saves it.
        :params tf.keras.models model: The model to save
        :params tf.keras.models history: the history to save
        :params int batch: Number of current iteration
        """
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
    """
    Main Function. A Model and next batch can be defined, if the fitting stopped at some point.
    Iterates through the batches and initiates the Training class.
    """
    config = conf.Config()
    batches = config.BATCHES
    model_path = None
    start_batch = 0
    for batch in range(batches):
        if batch < start_batch:
            continue
        print('Start training batch '+str(batch+1)+'/'+str(batches))
        tm = Train(batch, model_path)
        model_path = tm.run()
