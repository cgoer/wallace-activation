import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import matplotlib.pyplot as plt


class ModelTester:
    def __init__(self):
        pass

    def run(self):
        print('recording will start in 10 seconds')
        filename = 'data/generated/soundfiles/train/0/47.wav'
        sound = np.swapaxes(np.array(self.get_spectrogram(filename)), 0, 1)
        model = tf.keras.models.load_model('wallace_activation_batch1_12-04-2021_23-18-37')
        x = np.expand_dims(sound, axis=0)
        print(x.shape)
        predictions = model.predict(x)

        layers = model.layers
        layer_no = 15
        label_shape = layers[layer_no].output_shape
        label_shape = label_shape[1]

        label = self.convert_label(label_shape, np.load('data/generated/labels/train/0/47.npy'))
        label = np.swapaxes(np.array(label), 0, 1)
        label = np.reshape(label, (label_shape))

        plt.subplot(2, 1, 2)
        plt.plot(predictions[0,:,0],label='prediction')
        plt.plot(label,label='label', dashes=[6,2])
        plt.ylabel('Probability')
        plt.xlabel('Timestep')
        plt.legend()
        plt.show()

    @staticmethod
    def add_label(self, y, segment_end_ms):
        label_shape = y.shape[1]
        segment_end_y = int(segment_end_ms * label_shape / 10000)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < label_shape:
                y[0, i] = 1
        return y

    def convert_label(self, label_shape, label):
        converted_label = np.zeros((1, label_shape))
        for sequence in label:
            # add label at end time
            converted_label = self.add_label(converted_label, sequence[1])

        return converted_label

    @staticmethod
    def get_spectrogram(self, sound):
        rate, data = wavfile.read(sound)
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx


if __name__ == '__main__':
    sr = ModelTester()
    sr.run()
