import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import matplotlib.pyplot as plt


class ModelTester:
    def __init__(self):
        pass

    def run(self):
        filename = 'data/test1.wav'
        sound = np.swapaxes(np.array(self.get_spectrogram(filename)), 0, 1)
        model = tf.keras.models.load_model('models/wallace_activation_batch9_24-04-2021_15-54-50')
        x = np.expand_dims(sound, axis=0)
        print(x.shape)
        predictions = model.predict(x)

        layers = model.layers
        layer_no = 15
        label_shape = layers[layer_no].output_shape
        label_shape = label_shape[1]

        label = self.convert_label(label_shape, np.load('data/test1.npy'))
        label = np.swapaxes(np.array(label), 0, 1)
        label = np.reshape(label, (label_shape))

        plt.subplot(2, 1, 2)
        plt.plot(predictions[0,:,0],label='prediction')
        plt.plot(label,label='label', dashes=[6,2])
        plt.ylabel('Probability')
        plt.xlabel('Timestep')
        plt.legend()
        plt.show()

    def convert_label(self, label_shape, label):
        converted_label = np.zeros((1, label_shape))
        for sequence in label:
            # add label at end time
            converted_label = self.add_label(converted_label, sequence[0], sequence[1])

        return converted_label

    def add_label(self, y, segment_start_ms, segment_end_ms):
        label_shape = y.shape[1]
        segment_start_y = int(segment_start_ms * label_shape / 10000)
        segment_end_y = int(segment_end_ms * label_shape / 10000)
        for i in range(segment_start_y, segment_end_y):
            if i < label_shape:
                y[0, i] = 1
        return y

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
