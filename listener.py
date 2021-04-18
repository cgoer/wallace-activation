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
        self.width = config.RESPEAKER_WIDTH
        self.recording_chunk = config.RECORDING_CHUNK
        self.max_recording_time_s = config.MAX_RECORDING_S
        self.button = config.BUTTON_ID

        self.light = light.Lights()
        self.pyaudio = pyaudio.PyAudio()

        self.silence_threshold = 100
        self.chunk_duration = 0.5
        self.true_threshold = 0.5 #Predictions considered true
        self.muted = False
        self.recording_state = False
        self.sample_chunks = int(self.sample_rate * self.chunk_duration)
        self.clip_len_s = int(config.CLIP_LEN_MS/1000)
        self.feed_samples = int(self.sample_rate * self.clip_len_s)
        self.audio_format = self.pyaudio.get_format_from_width(self.width)

        self.interpreter, self.input_details, self.output_details = self.load_interpreter()

    def run(self):
        while True:
            if not self.muted:
                self.listen()
            self.check_for_mute_action()

    def listen(self):
        self.queue = Queue()
        self.data = np.zeros(self.feed_samples, dtype=self.audio_format)
        stream = self.get_stream()
        stream.start_stream()

        try:
            while not self.muted:
                self.data = self.queue.get()
                if self.check_for_keyword(self.get_spectrogram(self.data)):
                    self.after_keyword_action(stream)
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
        predictions = output_data[0][0]

        # remove the first few parts of the array due to weird prediction at the beginning
        predictions = predictions[50:]

        return np.maximum(predictions) > self.true_threshold


    def after_keyword_action(self, stream):
        self.recording_state = True
        frames = []
        silent_frames = 0
        self.light.listen()

        # Record until some silence was detected or the maximum rec time was reached
        for i in range(0, int(self.sample_rate / self.recording_chunk * self.max_recording_time_s)):
            data = stream.read(self.recording_chunk)
            frames.append(data)
            data_np = np.frombuffer(data, dtype=self.audio_format)
            if np.abs(data_np).mean() < self.silence_threshold:
                silent_frames += 1

            # If silence was detected for x timeframes end the recording
            if silent_frames >= int(self.sample_rate / self.recording_chunk * 1.5):
                break

        self.light.processing()

        # TODO: For now, save the file
        filename = 'command' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.wav'
        wf = wave.open('test/' + filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # TODO: Do some stuff with the recording (or whatever it will be later)
        # simulate some more processing time
        time.sleep(1)

        self.recording_state = False
        self.light.off()

    def callback(self, in_data, frame_count, time_info, status):
        # Skip callback in recording mode
        if self.recording_state:
            return (in_data, pyaudio.paContinue)

        data = np.frombuffer(in_data, dtype='int16')
        # return if there was no noise
        if np.abs(data).mean() < self.silence_threshold:
            return (in_data, pyaudio.paContinue)

        self.data = np.append(self.data, data)
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
            format=self.audio_format,
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
