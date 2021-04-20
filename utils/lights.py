from utils import apa102
import time


class Lights:
    LED_NO = 3

    def __init__(self):
        self.basis = [0] * 3 * self.LED_NO
        self.basis[0] = 2
        self.basis[3] = 1
        self.basis[4] = 1
        self.basis[7] = 2

        self.colors = [0] * 3 * self.LED_NO
        self.dev = apa102.APA102(num_led=self.LED_NO)

    def off(self):
        self.set([0] * 3 * self.LED_NO)

    def set(self, colors):
        for i in range(self.LED_NO):
            self.dev.set_pixel(i, int(colors[3 * i]), int(colors[3 * i + 1]), int(colors[3 * i + 2]))
        self.dev.show()

    def listen(self):
        colors = [2, 0, 0, 1, 1, 0, 0, 2, 0]
        self.set(colors)

    def processing(self):
        colors = [2, 0, 0, 1, 1, 0, 0, 2, 0]
        self.set(colors)


    def mute(self):
        colors = [2, 0, 0, 1, 1, 0, 0, 2, 0]
        self.set(colors)