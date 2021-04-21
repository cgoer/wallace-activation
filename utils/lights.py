from utils import apa102


class Lights:
    LED_NO = 3

    def __init__(self):
        self.BLUE = [0, 0, 2, 0, 0, 2, 0, 0, 2]
        self.RED = [2, 0, 0, 2, 0, 0, 2, 0, 0]
        self.GREEN = [0, 2, 0, 0, 2, 0, 0, 2, 0]
        self.BWB = [0, 0, 2, 2, 2, 2, 0, 0, 2]
        self.max_brightness = 75
        self.dev = apa102.APA102(num_led=self.LED_NO)

    def off(self):
        self.set([0] * 3 * self.LED_NO)

    def set(self, colors):
        for i in range(self.LED_NO):
            self.dev.set_pixel(i, int(colors[3 * i]), int(colors[3 * i + 1]), int(colors[3 * i + 2]), self.max_brightness)
        self.dev.show()

    def listen(self):
        self.set(self.GREEN)

    def processing(self):
        self.set(self.BLUE)

    def mute(self):
        self.set(self.RED)
