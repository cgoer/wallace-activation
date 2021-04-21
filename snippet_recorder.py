import RPi.GPIO as GPIO
import time
import pyaudio
import wave
import utils.config as conf
import utils.lights as lit
from datetime import datetime


class SnippetRecorder:
    """
    Record a snippet with the Seeed RaspberryPi Hat.
    Start the script and press the button to start recording.
    Recording starts when the Lights turn on and stops when they turn off again.
    """

    def __init__(self, record_seconds):
        # Setup config params
        config = conf.Config()
        self.sample_rate = config.WAV_FRAMERATE_HZ
        self.button = config.BUTTON_ID
        self.channels = config.RESPEAKER_CHANNELS
        self.width = config.RESPEAKER_WIDTH
        self.index = config.RESPEAKER_INDEX
        self.chunk_size = config.RECORDING_CHUNK

        self.RECORD_SECONDS = record_seconds

        # init button
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.button, GPIO.IN)

        # init lights
        self.light = lit.Lights()
        self.light.off()

    def wait_for_action(self):
        """
        Wait for button press to start recording. Endless loop.
        """
        while True:
            state = GPIO.input(self.button)
            if not state:
                print('recording will start in 1 second')
                time.sleep(1)
                self.record()
            else:
                print('Press the Button to record')
            time.sleep(1)

    def record(self):
        """
        Record sound and save in a file.
        :return:
        """
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

        for i in range(0, int(self.sample_rate / self.chunk_size * self.RECORD_SECONDS)):
            data = stream.read(self.chunk_size)
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
    record_seconds = 1
    sr = SnippetRecorder(record_seconds)
    sr.wait_for_action()
