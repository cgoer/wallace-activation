import RPi.GPIO as GPIO
import time
import pyaudio
import wave
import utils.config as conf
import os
import utils.lights as lit
from datetime import datetime


class SampleRecorder:
    CHUNK = 1024
    RECORD_SECONDS = 1

    def __init__(self):
        # Setup config params
        config = conf.Config()
        self.sample_rate = config.WAV_FRAMERATE_HZ
        self.button = config.BUTTON_ID
        self.channels = config.RESPEAKER_CHANNELS
        self.width = config.RESPEAKER_WIDTH
        self.index = config.RESPEAKER_INDEX

        # init button
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.button, GPIO.IN)

        # init lights
        self.light = lit.Lights()
        self.light.off()

        # FIRE!
        self.wait_for_action()

    def wait_for_action(self):
        while True:
            state = GPIO.input(self.button)
            if not state:
                print('recording will start in 1 second')
                time.sleep(1)
                self.record()
            else:
                print('Press the Button to record')
            time.sleep(0.5)

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            rate=self.sample_rate,
            format=p.get_format_from_width(self.width),
            channels=self.channels,
            input=True,
            input_device_index=self.index)

        print("* recording")
        self.light.listen()

        frames = []

        for i in range(0, int(self.sample_rate / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording")
        self.light.off()

        stream.stop_stream()
        stream.close()
        p.terminate()

        filename = 'recording' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.wav'
        wf = wave.open('recordings/' + filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(self.width)))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        print('Saved file ' + filename)
        time.sleep(1)


if __name__ == '__main__':
    SampleRecorder()
