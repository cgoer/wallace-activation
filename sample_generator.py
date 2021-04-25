import numpy as np
from pydub import AudioSegment
import random
import os
import utils.config as conf

class SampleGenerator:
    """
    Sample Generator generates Sample files of a certain length from a set of backgrounds, keywords and non_keywords.
    The imput data has to be in the directories defined at utils/config.py RAW_SOUND_DATA_PATHS constant.
    Input data may have .mp3 or .wav format (IMPORTANT: .mp3 files will be deleted after conversion to .wav!)
    Input data may have any sample rate.
    Background input data may be any length, only the first x seconds will be imported.
    Non-Keyword data may be any length, only the first x seconds will be imported (see self.import_data() for more info)
    This script will output a set of training and test data in the directory defined at utils/config.py GENERATED_DATA_PATHS constant.
    It will also output an array of labels to each generated file. Each label containing arrays of the Begin and end time (ms) of each keyword in a sound file.
    """
    def __init__(self):
        # Import & Set Configuration
        config = conf.Config()
        self.context_paths = config.CONTEXT_PATHS
        self.import_data_paths = config.RAW_SOUND_DATA_PATHS
        self.export_data_paths = config.GENERATED_DATA_PATHS
        self.framerate = config.WAV_FRAMERATE_HZ
        self.clip_len_ms = config.CLIP_LEN_MS
        self.training_split = config.TRAINING_SPLIT_PERCENT
        self.batches = config.BATCHES
        self.batch_size = config.BATCH_SIZE
        
        # Add Directories if missing
        self.add_missing_directories()

        # Convert mp3 Files to wav
        for keyword in self.import_data_paths:
            self.convert_all_to_wav(self.import_data_paths[keyword])

    def run(self):
        """
        Generate x datasets. Load, convert, resample and split Data before generating.
        """
        print('Loading Data')
        keyword, non_keyword, background = self.load_data(self.import_data_paths)
        print('Found Datasets:')
        print('Keywords: ' + str(len(keyword)))
        print('Non-Keywords: ' + str(len(non_keyword)))
        print('Backgrounds: ' + str(len(background)))
        print('----------')
        keyword_train, keyword_test = self.split_train_test(keyword)
        non_keyword_train, non_keyword_test = self.split_train_test(non_keyword)
        background_train, background_test = self.split_train_test(background)

        for i in range(self.batches):
            runs = int(self.batch_size*(self.training_split/100))
            self.generate_samples('train', i, runs, background_train, keyword_train, non_keyword_train)

            runs = int(self.batch_size*(1-self.training_split/100))
            self.generate_samples('test', i, runs, background_test, keyword_test, non_keyword_test)

        print('done.')

    def generate_samples(self, run_type, batch, runs, backgrounds, keywords, non_keywords):
        """
        Generate x samples of randomly selected backgrounds, keywords and non_keywords.
        :param str run_type: Information about what type of sample set is generated (train/test)
        :param int runs: Number of runs
        :param list backgrounds: list of background AudioSegment objects
        :param list keywords: list of keyword AusioSegment objects
        :param list non_keywords: list of non_keyword AudioSegment objects
        """
        print('Processing ' + run_type + ' Data')
        for run_no in range(runs):
            run_no += 1
            print(str(run_no) + '/' + str(runs))
            self.generate_and_save_sample(backgrounds, keywords, non_keywords,
                                                              run_type, run_no, batch)

    def generate_and_save_sample(self, backgrounds, keywords, non_keywords, run_type, run_no, batch):
        """
        Pick a random background and randomly place keywords and non-keywords onto it.
        Saves the generated sample and labels into desired directories.
        :param list backgrounds:
        :param list keywords:
        :param list non_keywords:
        :param str run_type:
        :param int run_no:
        :param int batch:
        """
        # Choose background and lower volume
        background = random.choice(backgrounds) - 25
        
        label = []
        indices = np.random.randint(len(keywords), size=10)
        keyword_set = [keywords[i] for i in indices]
        indices = np.random.randint(len(non_keywords), size=10)
        non_keyword_set = [non_keywords[i] for i in indices]

        # Pick KW and NKW Samples
        order = np.random.choice([0, 1], 10, True, [0.5, 0.5])

        time_left = self.clip_len_ms
        keyword_count = 0
        non_keyword_count = 0
        
        # Fill the background clip with chosen samples
        while time_left and (keyword_count + non_keyword_count) < len(order):
            # Random pause
            time_between_clips = np.random.randint(500, 1500)
            
            # Stop if no time left
            if time_left < time_between_clips:
                break

            start = (self.clip_len_ms - time_left) + time_between_clips
            if order[(keyword_count + non_keyword_count)] == 0:
                background, time_left, did_write = self.add_clip_to_background(background,
                                                    non_keyword_set[non_keyword_count], time_left, time_between_clips)
                non_keyword_count += 1
            else:
                background, time_left, did_write = self.add_clip_to_background(background, keyword_set[keyword_count],
                                                                               time_left, time_between_clips)
                keyword_count += 1
                if did_write:
                    label.append([start,(self.clip_len_ms - time_left)])

            # If last word wasn't written, use next word from same pool (if its not the last one already)
            if (did_write is False) and ((keyword_count + non_keyword_count + 1) < len(order)):
                order[(keyword_count + non_keyword_count)+1] = order[(keyword_count + non_keyword_count)]

        # Export sound File
        file_path = self.export_data_paths['sound'] + run_type + '/' + str(batch) + '/' + str(run_no) + ".wav"
        background.export(file_path, format="wav")

        # Export label
        np.save(self.export_data_paths['label'] + self.context_paths[run_type] + str(batch) + '/' + str(run_no) + '.npy', np.array(label))

    def add_clip_to_background(self, background, sound_snippet, time_left, time_between_clips):
        """
        Lay a sound snippet over a background sound at a desired position.
        Return the new background sound, the time left at the end of the bg sound and a bool if the snippet was written.
        :param AudioSegment background: Background Sound Object
        :param AudioSegment sound_snippet: Sound snippet Object to add to Background
        :param int time_left: Time left to the end of Sound Object
        :param int time_between_clips: Length of pause to last clip
        :returns AudioSegment background, int time_left, bool did_write:
        """
        segment_ms = len(sound_snippet)

        start = (self.clip_len_ms - time_left) + time_between_clips

        did_write = False

        if ((start + segment_ms) < self.clip_len_ms):
            time_left = time_left - (segment_ms + time_between_clips)
            background = background.overlay(sound_snippet, position=start)
            did_write = True
        return background, time_left, did_write

    def load_data(self, paths):
        """
        Loads data from certain filepath. Resamples files automatically.
        :param dictionary paths: filepaths to load data from
        :return list keyword: list of AudioSegment objects
        :return list non_keyword: list of AudioSegment objects
        :return list background: list of AudioSegment objects
        """
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
                nk = self.resample(AudioSegment.from_wav(paths['non_keyword'] + filename))
                # only take up to 4s
                non_keyword.append(nk[:4000])
        return (keyword, non_keyword, background)

    def convert_all_to_wav(self, directory):
        """
        Check a direcotry for .mp3 files and convert them to .wav. Delete .mp3 version afterwards.
        :param str directory: filepath
        """
        print('Converting .mp3 files to .wav')
        for filename in os.listdir(directory):
            if filename.endswith('mp3'):
                sound = AudioSegment.from_mp3(directory + filename)
                sound.export(directory + os.path.splitext(filename)[0] + '.wav', format="wav")
                os.remove(directory + filename)

    def resample(self, file):
        """
        Resample File to default framerate if needed
        :param file: AudioSegment object
        :return file: AudioSegment object
        """
        if (file.frame_rate != self.framerate):
            return file.set_frame_rate(self.framerate)
        return file

    def add_missing_directories(self):
        """
        Check if all necessary directories are present. Exit if there are no data directories.
        """
        for export_dirs in self.export_data_paths:
            for contexts in self.context_paths:
                for i in range(self.batches):
                    if not os.path.exists(self.export_data_paths[export_dirs] + self.context_paths[contexts] + '/' + str(i)):
                        os.makedirs(self.export_data_paths[export_dirs] + self.context_paths[contexts]+ '/' + str(i))

        for import_dirs in self.import_data_paths:
            if not os.path.exists(self.import_data_paths[import_dirs]):
                os.makedirs(self.import_data_paths[import_dirs])
                print('Import Directory ' + self.import_data_paths[import_dirs] + ' was missing.')
                print('Directory Added. Please fill with data! Aborting.')
                exit(404)

    def split_train_test(self, sound_array):
        """
        Randomly split array into train and test array.
        :param sound_array: list of AudioSegment objects
        :return train, test: list of AudioSegment objects
        """
        random.shuffle(sound_array)
        train_len = int(len(sound_array)*(self.training_split/100))
        train, test = sound_array[:train_len], sound_array[train_len:]
        return train, test


if __name__ == '__main__':
    sg = SampleGenerator()
    sg.run()
