from apa102_pi.driver import apa102


class Lights:
    def __init__(self):
        self.strip = apa102.APA102(num_led=3, mosi=10, sclk=11, order='rbg')
        self.strip.set_global_brightness(15)

        self.BLUE = 0x0000FF
        self.RED = 0xFF0000
        self.WHITE = 0x000000

    def off(self):
        self.strip.clear_strip()
        self.strip.cleanup()

    def listen(self):
        self.strip.set_pixel_rgb(1, self.BLUE)
        self.strip.set_pixel_rgb(2, self.BLUE)
        self.strip.set_pixel_rgb(3, self.BLUE)
        self.strip.show()

    def processing(self):
        self.strip.set_pixel_rgb(1, self.WHITE)
        self.strip.set_pixel_rgb(2, self.BLUE)
        self.strip.set_pixel_rgb(3, self.WHITE)
        self.strip.show()

    def mute(self):
        self.strip.set_pixel_rgb(1, self.RED)
        self.strip.set_pixel_rgb(2, self.RED)
        self.strip.set_pixel_rgb(3, self.RED)
        self.strip.show()
