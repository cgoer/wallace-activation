import RPi.GPIO as GPIO
import time
import pyaudio
import wave
import config as conf
    
class SampleRecorder:
    BUTTON = 17
    RESPEAKER_CHANNELS = 1
    RESPEAKER_WIDTH = 2
    # run getDeviceInfo.py to get index
    RESPEAKER_INDEX = 2  # refer to input device id
    CHUNK = 1024
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "output.wav"

    def __init__(self):
        # Setup config params
        config = conf.Config()
        self.sample_rate = config.WAV_FRAMERATE_HZ

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BUTTON, GPIO.IN)
        self.wait_for_action()

    def wait_for_action(self):
        while True:
            state = GPIO.input(self.BUTTON)
            if state:
                print("off")
            else:
                print("on")
            time.sleep(1)

    def stuff(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            rate=self.sample_rate,
            format=p.get_format_from_width(self.RESPEAKER_WIDTH),
            channels=self.RESPEAKER_CHANNELS,
            input=True,
            input_device_index=self.RESPEAKER_INDEX, )

        print("* recording")

        frames = []

        for i in range(0, int(self.sample_rate / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.RESPEAKER_CHANNELS)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(self.RESPEAKER_WIDTH)))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

if __name__ == '__main__':
    SampleRecorder()
