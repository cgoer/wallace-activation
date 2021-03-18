import numpy as np
from pydub import AudioSegment
import random
import os
from scipy.io import wavfile
import matplotlib.pyplot as pyplt
import config as conf

# TODO: document
class SampleGenerator:
    def __init__(self, check_directories, convert_to_wav):
        # Import & Set Configuration
        config = conf.Config()
        self.context_paths = config.CONTEXT_PATHS
        self.import_data_paths = config.RAW_SOUND_DATA_PATHS
        self.export_data_paths = config.GENERATED_DATA_PATHS
        self.framerate = config.WAV_FRAMERATE_HZ
        self.ty = config.TY
        self.clip_len_ms = config.CLIP_LEN_MS
        self.training_split = config.TRAINING_SPLIT_PERCENT
        self.seed = config.SEED
        
        # Add Directories if missing
        if check_directories:
            self.add_missing_directories()

        # Convert mp3 Files to wav
        if convert_to_wav:
            for keyword in self.import_data_paths:
                self.convert_all_to_wav(self.import_data_paths[keyword])

        # Set Seed
        np.random.seed(self.seed)

        # Fire
        self.generate_samples()

    def generate_samples(self):
        keyword, non_keyword, background = self.load_data(self.import_data_paths)
        keyword_train, keyword_test = self.split_train_test(keyword)
        non_keyword_train, non_keyword_test = self.split_train_test(non_keyword)
        background_train, background_test = self.split_train_test(background)

        # TODO: clean up this mess..
        run_no = 1
        run_type = 'train'
        spectrograms = []
        labels = []
        for x in range(5):
            spectrogram, label = self.create_training_example(background_train, keyword_train, non_keyword_train, run_type, run_no)
            spectrograms.append(spectrogram)
            labels.append(label)
            run_no += 1
        spectrograms_numpy = np.array(spectrograms)
        labels_numpy = np.array(labels)
        np.save(self.export_data_paths['numpy']+self.context_paths[run_type]+'spectrograms.npy', spectrograms_numpy)
        np.save(self.export_data_paths['numpy']+self.context_paths[run_type]+'labels.npy', labels_numpy)

        run_no = 1
        run_type = 'test'
        spectrograms = []
        labels = []
        for x in range(5):
            spectrogram, label = self.create_training_example(background_test, keyword_test, non_keyword_test, run_type, run_no)
            spectrograms.append(spectrogram)
            labels.append(label)
            run_no += 1
        spectrograms_numpy = np.array(spectrograms)
        labels_numpy = np.array(labels)
        np.save(self.export_data_paths['numpy']+self.context_paths[run_type]+'spectrograms.npy', spectrograms_numpy)
        np.save(self.export_data_paths['numpy']+self.context_paths[run_type]+'labels.npy', labels_numpy)
        print('done.')
        
    def insert_ones(self, y, segment_end_ms):
        segment_end_y = int(segment_end_ms * self.ty / self.clip_len_ms)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < self.ty:
                y[0, i] = 1
        return y

    def insert_clip(self, background, random_activate, time_left, time_between_clips):
        segment_ms = len(random_activate)

        start = self.clip_len_ms - time_left
        start = start + time_between_clips

        if ((start + segment_ms) < self.clip_len_ms):
            time_left = time_left - (segment_ms + time_between_clips)

            background = background.overlay(random_activate, position=start)
        return background, time_left

    def create_training_example(self, backgrounds, activates, negatives, run_type, run_no):
        background = random.choice(backgrounds) - 20
        label = np.zeros((1, self.ty))
        random_indices = np.random.randint(len(activates), size=5)
        random_activates = [activates[i] for i in random_indices]

        random_indices = np.random.randint(len(negatives), size=5)
        random_negatives = [negatives[i] for i in random_indices]
        random_order = np.random.randint(0,2,(5))

        time_left = self.clip_len_ms
        keyword_count = 0
        non_keyword_count = 0
        while time_left and (keyword_count + non_keyword_count) < len(random_order):
            time_between_clips = np.random.randint(500, 3000)
            if random_order[(keyword_count + non_keyword_count)] == 0:
                background, time_left = self.insert_clip(background, random_negatives[non_keyword_count], time_left, time_between_clips)
                non_keyword_count += 1
            else:
                background, time_left = self.insert_clip(background, random_activates[keyword_count], time_left, time_between_clips)
                label = self.insert_ones(label, self.clip_len_ms - time_left)
                keyword_count += 1

        background = self.match_target_amplitude(background, -20.0)
        file_path = self.export_data_paths['sound'] + run_type + '/' + str(run_no) + ".wav"
        file_handle = background.export(file_path, format="wav")
        spectrogram = self.graph_spectrogram(file_path)

        return spectrogram, label

    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def get_wav_info(self, wav_file):
        rate, data = wavfile.read(wav_file)
        return rate, data

    def graph_spectrogram(self, wav_file):
        rate, data = self.get_wav_info(wav_file)
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = pyplt.specgram(data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = pyplt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx

    def load_data(self, paths):
        keyword = []
        background = []
        non_keyword = []
        for filename in os.listdir(paths['keyword']):
            if filename.endswith("wav"):
                keyword.append(self.resample(AudioSegment.from_wav(paths['keyword'] + filename)))
        for filename in os.listdir(paths['background']):
            if filename.endswith("wav"):
                background_sound = self.resample(AudioSegment.from_wav(paths['background'] + filename))
                # Take only first x milliseconds
                background.append(background_sound[:self.clip_len_ms])
        for filename in os.listdir(paths['non_keyword']):
            if filename.endswith("wav"):
                non_keyword.append(self.resample(AudioSegment.from_wav(paths['non_keyword'] + filename)))
        return (keyword, non_keyword, background)

    def convert_all_to_wav(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith('mp3'):
                sound = AudioSegment.from_mp3(directory + filename)
                sound.export(directory + os.path.splitext(filename)[0] + '.wav', format="wav")
                os.remove(directory + filename)

    def resample(self, file):
        """
        Resamples File to default framerate if needed
        :param file: AudioSegment object
        :return: AudioSegment object
        """
        if (file.frame_rate != self.framerate):
            return file.set_frame_rate(self.framerate)
        return file

    def add_missing_directories(self):
        for export_dirs in self.export_data_paths:
            for contexts in self.context_paths:
                if not os.path.exists(self.export_data_paths[export_dirs] + self.context_paths[contexts]):
                    os.makedirs(self.export_data_paths[export_dirs] + self.context_paths[contexts])

        for import_dirs in self.import_data_paths:
            if not os.path.exists(self.import_data_paths[import_dirs]):
                os.makedirs(self.import_data_paths[import_dirs])
                print('Import Directory ' + self.import_data_paths[import_dirs] + ' was missing.')
                print('Directory Added. Please fill with data! Aborting.')
                exit(404)

    def split_train_test(self, sound_array):
        random.shuffle(sound_array)
        train_len = int(len(sound_array)*(self.training_split/100))
        train, test = sound_array[:train_len], sound_array[train_len:]
        return train, test


if __name__ == '__main__':
    convert_mp3_to_wav = True
    check_and_add_directories = True
    SampleGenerator(check_and_add_directories, convert_mp3_to_wav)
