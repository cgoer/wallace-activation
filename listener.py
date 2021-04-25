import numpy as np
import matplotlib.pyplot as plt
import utils.config as conf
import wave
import pyaudio
import time
from queue import Queue
from tflite_runtime.interpreter import Interpreter
from datetime import datetime


class Listener:
    """
    Listens to input stream and calls a prediction class to classify if a certain keyword was said.
    Records sound if keyword was said.
    """
    def __init__(self, raspi_mode):
        """
        :param boolean raspi_mode: If true, loads raspberry-specific libraries and enables specific methods
        """
        self.raspi_mode = raspi_mode

        # Load RPi specific Modules
        if self.raspi_mode:
            import RPi.GPIO as GPIO
            import utils.lights as light

        # Load config params
        config = conf.Config()
        self.sample_rate = config.WAV_FRAMERATE_HZ
        self.channels = config.RESPEAKER_CHANNELS if self.raspi_mode else config.MAC_CHANNELS
        self.recording_chunk = config.RECORDING_CHUNK
        self.max_recording_frames = config.MAX_RECORDING_FRAMES
        self.button = config.BUTTON_ID
        self.width = config.RESPEAKER_WIDTH if self.raspi_mode else config.MAC_WIDTH
        self.format = config.RESPEAKER_FORMAT if self.raspi_mode else config.MAC_FORMAT
        self.index = config.RESPEAKER_INDEX if self.raspi_mode else config.MAC_INDEX
        self.max_silent_frames = config.MAX_SILENT_FRAMES

        self.pyaudio = pyaudio.PyAudio()

        # init button & Lights
        if self.raspi_mode:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.button, GPIO.IN)
            self.gpio = GPIO
            self.light = light.Lights()

        self.silence_threshold = 32.5
        self.chunk_duration = 0.7
        self.sample_chunks = int(self.sample_rate * self.chunk_duration)
        self.clip_len_s = int(config.CLIP_LEN_MS / 1000)
        self.feed_samples = int(self.sample_rate * self.clip_len_s)

        # state vars
        self.muted = False
        self.recording_state = False
        self.silent_frames = 0
        self.recorded_frames = 0
        self.data = np.zeros(self.feed_samples, dtype=self.format)
        self.recording = []

        self.queue = Queue()
        self.Predictor = Predictor()

    def run(self):
        """
        Main Method. Infinite loop: calls listen method if mute state is false.
        """
        while True:
            if not self.muted:
                self.listen()
            self.check_for_mute_action()

    def listen(self):
        """
        Activates input stream and calls prediction class if there is input to validate. Infinite loop.
        """
        print('Start listening')
        if self.raspi_mode:
            self.light.off()

        stream = self.get_stream()
        stream.start_stream()

        try:
            while not self.muted:
                if self.recording_state:
                    continue
                data = self.queue.get()
                if len(data) > 0 and self.Predictor.predict_data(data):
                    self.on_keyword()
                self.check_for_mute_action()
        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()
            exit(1)

        stream.stop_stream()
        stream.close()

    def check_for_mute_action(self):
        """
        check if button was pressed. Toggles mute state and performs light changes.
        """
        if self.raspi_mode:
            state = self.gpio.input(self.button)
        else:
            return

        if state:
            return

        self.muted = not self.muted
        if self.muted:
            self.light.mute()
            time.sleep(1)


    def on_keyword(self):
        """
        Perform actions if Keyword was said
        """
        self.recorded_frames = 0
        self.silent_frames = 0
        self.recording_state = True
        self.queue.empty()
        print('recording')
        if self.raspi_mode:
            self.light.listen()

    def on_command(self, data):
        """
        Perform actions after voice input was recorded.
        :param np.array data: The voice input data
        """
        self.recording_state = False

        # Simulate action
        time.sleep(5)

        if self.raspi_mode:
            self.light.processing()

        # TODO: For now, save the file
        filename = 'testapp' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.wav'
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.get_sample_size(pyaudio.get_format_from_width(self.width)))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(data))
        wf.close()

        self.recording = []
        self.data = np.zeros(self.feed_samples, dtype=self.format)
        self.queue.empty()
        self.recording_state = False

        if self.raspi_mode:
            self.light.off()

    def callback(self, in_data, frame_count, time_info, status):
        """
        Callback called from audio stream. If certain volume thereshold is overridden, saves data into queue for processing.
        Saves data in different variable while recording. Ends recording state if certain silence thereshold or timespan passed.
        """
        data = np.frombuffer(in_data, dtype=self.format)
        if self.recording_state:
            self.recorded_frames += 1
            self.recording = np.append(self.recording, in_data)
            if (np.abs(data).mean() < self.silence_threshold):
                self.silent_frames += 1
            
            if (self.recorded_frames >= self.max_recording_frames) or self.silent_frames > self.max_silent_frames:
                print('recorded frames:' + str(self.recorded_frames))
                print('silent frames:' + str(self.silent_frames))
                self.on_command(self.recording)
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

    def get_stream(self):
        """
        Open pyaudio stream
        :returns pyaudio: Stream object
        """
        stream = self.pyaudio.open(
            format=pyaudio.get_format_from_width(self.width),
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.sample_chunks,
            input_device_index=self.index,
            stream_callback=self.callback)
        return stream


class Predictor:
    """
    Predicts provided sound data. Loads tflite model.
    """
    def __init__(self):
        self.true_threshold = 0.8 #Predictions considered true
        self.interpreter, self.input_details, self.output_details = self.load_interpreter()

    def predict_data(self, data):
        """
        Predicts provided sound data using a previously loaded tflite model
        :param np.array data: np.array of sound data
        :returns boolean: Boolean if keyword was found or not
        """
        spectrogram = self.get_spectrogram(data)
        spectrogram = np.float32(np.expand_dims(spectrogram.swapaxes(0, 1), axis=0))
        self.interpreter.set_tensor(self.input_details[0]['index'], spectrogram)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = np.reshape(output_data, -1)

        # remove the first few parts of the array due to weird prediction at the beginning
        predictions = predictions[50:]
        print(np.amax(predictions))

        return np.amax(predictions) > self.true_threshold

    @staticmethod
    def load_interpreter():
        """
        Loads tflite model
        :returns: Model interpreter object, input details and output details
        """
        # TODO: Remove static model
        interpreter = Interpreter('models/wallace_activation_batch9_24-04-2021_15-54-50.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    @staticmethod
    def get_spectrogram(data):
        """
        Returns spectrogram from giben sound data array
        :param np.array data: np.array of sound data
        :returns np.array: Spectrogram as np.array
        """
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            pxx, _, _, _ = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx


if __name__ == '__main__':
    """
    Runs listener application. 
    Set Param to True if runing on RaspberryPi
    """
    listener = Listener(False)
    listener.run()
