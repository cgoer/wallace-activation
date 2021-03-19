import RPi.GPIO as GPIO
import time
import pyaudio
import wave
import config as conf
import os
from datetime import datetime
import apa102


class SampleRecorder:
    BUTTON = 17
    RESPEAKER_CHANNELS = 1
    RESPEAKER_WIDTH = 2
    RESPEAKER_INDEX = 1  # refer to input device id
    CHUNK = 1024
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "output.wav"

    PIXELS_N = 3

    def test(self):
        self.basis = [0] * 3 * self.PIXELS_N
        self.basis[0] = 2
        self.basis[3] = 1
        self.basis[4] = 1
        self.basis[7] = 2

        self.colors = [0] * 3 * self.PIXELS_N
        self.dev = apa102.APA102(num_led=self.PIXELS_N)

    def _off(self):
        self.write([0] * 3 * self.PIXELS_N)

    def write(self, colors):
        for i in range(self.PIXELS_N):
            self.dev.set_pixel(i, int(colors[3 * i]), int(colors[3 * i + 1]), int(colors[3 * i + 2]))

        self.dev.show()

    def __init__(self):
        # Setup config params
        config = conf.Config()
        self.sample_rate = config.WAV_FRAMERATE_HZ

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BUTTON, GPIO.IN)
        self.test()
        self._off()
        self.wait_for_action()

    def _listen(self):
        for i in range(1, 25):
            colors = [i * v for v in self.basis]
            self.write(colors)
            time.sleep(0.01)

    def wait_for_action(self):
        while True:
            state = GPIO.input(self.BUTTON)
            if not state:
                print('recording will start in 1 second')
                time.sleep(1)
                self.stuff()
            else:
                print('Press the Button to record')
            time.sleep(0.5)

    def stuff(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            rate=self.sample_rate,
            format=p.get_format_from_width(self.RESPEAKER_WIDTH),
            channels=self.RESPEAKER_CHANNELS,
            input=True,
            input_device_index=self.RESPEAKER_INDEX)

        print("* recording")
        self._listen()

        frames = []

        for i in range(0, int(self.sample_rate / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording")
        self._off()

        stream.stop_stream()
        stream.close()
        p.terminate()

        filename = 'recording' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.wav'
        wf = wave.open('recordings/' + filename, 'wb')
        wf.setnchannels(self.RESPEAKER_CHANNELS)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(self.RESPEAKER_WIDTH)))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        print('Saved file ' + filename)
        time.sleep(1)


if __name__ == '__main__':
    SampleRecorder()
