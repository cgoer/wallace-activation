import RPi.GPIO as GPIO
import numpy as np
import matplotlib.pyplot as plt
import utils.config as conf
import utils.lights as light
import wave
import pyaudio
import time
from queue import Queue
from tflite_runtime.interpreter import Interpreter
from datetime import datetime

class Listener:
    def __init__(self):
        # Load config params
        config = conf.Config()
        self.sample_rate = config.WAV_FRAMERATE_HZ
        self.channels = config.RESPEAKER_CHANNELS
        self.recording_chunk = config.RECORDING_CHUNK
        self.max_recording_time_s = config.MAX_RECORDING_S
        self.button = config.BUTTON_ID
        self.width = config.RESPEAKER_WIDTH

        self.light = light.Lights()
        self.pyaudio = pyaudio.PyAudio()

        # init button
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.button, GPIO.IN)

        self.silence_threshold = 32.5
        self.chunk_duration = 0.5
        self.true_threshold = 0.5 #Predictions considered true
        self.muted = False
        self.recording_state = False
        self.silent_frames = 0
        self.recorded_frames = 0
        self.sample_chunks = int(self.sample_rate * self.chunk_duration)
        self.clip_len_s = int(config.CLIP_LEN_MS/1000)
        self.feed_samples = int(self.sample_rate * self.clip_len_s)

        self.queue = Queue()
        self.data = np.zeros(self.feed_samples, dtype='int8')
        self.recording = []

        self.interpreter, self.input_details, self.output_details = self.load_interpreter()

    def run(self):
        while True:
            if not self.muted:
                self.listen()
            self.check_for_mute_action()

    def listen(self):
        print('listening')
        self.light.off()


        stream = self.get_stream()
        stream.start_stream()

        try:
            while not self.muted:
                if self.recording_state:
                    continue
                data = self.queue.get()
                if len(data) > 0 and self.check_for_keyword(self.get_spectrogram(data)):
                    self.after_keyword_action()
                self.check_for_mute_action()
        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()
            exit(1)

        stream.stop_stream()
        stream.close()

    def check_for_mute_action(self):
        state = GPIO.input(self.button)
        # return if button was not pressed
        if state:
            return

        self.muted = not self.muted
        if self.muted:
            self.light.mute()

    def check_for_keyword(self, spectrogram):
        spectrogram = np.float32(np.expand_dims(spectrogram.swapaxes(0, 1), axis=0))
        self.interpreter.set_tensor(self.input_details[0]['index'], spectrogram)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = np.reshape(output_data, -1)

        # remove the first few parts of the array due to weird prediction at the beginning
        predictions = predictions[50:]

        return predictions > self.true_threshold


    def after_keyword_action(self):
        self.recorded_frames = 0
        self.silent_frames = 0
        self.recording_state = True
        self.queue.empty()
        self.light.listen()


    def after_recording(self, data):
        self.recording_state = False
        time.sleep(5)

        self.light.processing()

        # TODO: For now, save the file
        filename = 'testapp' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.wav'
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.get_sample_size(pyaudio.get_format_from_width(self.width)))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(data))
        wf.close()

        self.recording_state = False
        self.recording = []
        self.data = np.zeros(self.feed_samples, dtype='int8')

        self.light.off()

    def callback(self, in_data, frame_count, time_info, status):
        data = in_data
        if self.recording_state:
            self.recording = np.append(self.recording, in_data)
            self.recorded_frames += 1
            if (np.abs(data).mean() < self.silence_threshold):
                self.silent_frames += 1
            if (self.recorded_frames >= self.max_recording_time_s) or self.silent_frames > 2:
                print('recorded frames:' + str(self.recorded_frames))
                print('silent frames:' + str(self.silent_frames))
                self.after_recording(self.recording)
            return (in_data, pyaudio.paContinue)

        # return if there was no noise
        if np.abs(data).mean() < self.silence_threshold:
            print('silence')
            return (in_data, pyaudio.paContinue)

        self.data = np.append(self.data, data)
        print('y')
        if len(self.data) > self.feed_samples:
            self.data = self.data[-self.feed_samples:]
            self.queue.put(self.data)
        return (in_data, pyaudio.paContinue)

    @staticmethod
    def load_interpreter():
        interpreter = Interpreter('models/wallace_activation_batch1_12-04-2021_23-18-37.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    @staticmethod
    def get_spectrogram(data):
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            pxx, _, _, _ = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx

    def get_stream(self):
        stream = self.pyaudio.open(
            format=pyaudio.get_format_from_width(self.width),
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.sample_chunks,
            input_device_index=0,
            stream_callback=self.callback)
        return stream


if __name__ == '__main__':
    listener = Listener()
    listener.run()
