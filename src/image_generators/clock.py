from general_image_generator import ImageGenerator
import math
import random
import numpy as np
from PIL import ImageDraw, Image

class Clock(ImageGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.n_layers   = int(config['CLOCK']['n_hands'])
        self.hand_color = int(config['CLOCK']['hand_color'])
        self.hand_width = int(config['CLOCK']['hand_width']) * self.antialias_factor
        self.img_size = self.target_size * self.antialias_factor

    def sample(self, n):
        self.logger.info('Generating ' + str(n) + ' clock images...')
        samples = []
        for i in range(n):
            samples.append(self.__create_image())
        samples = np.array(samples)

        self.logger.debug('Final shape: ' + str(samples.shape))
        self.logger.info('Done.')

        return samples


    def __create_2d_image(self):
        img_obj = Image.new("L", (self.img_size, self.img_size), 0)
        img_draw = ImageDraw.Draw(img_obj)

        # Drawing in random angle
        angle = self.__random_angle()
        img_draw.line(self.__get_clock_hand_coords(angle), self.hand_color, self.hand_width)
        
        # Resize with some blurring around the edges
        img_obj = img_obj.resize((self.target_size, self.target_size), Image.ANTIALIAS)
        
        return np.array(img_obj)

    def __create_image(self):
        img_layers = []
        for layer_i in range(self.n_layers):
            img_layers.append(self.__create_2d_image())
        img_layers = np.array(img_layers)

        assert img_layers.shape == (self.n_layers, self.target_size, self.target_size), img_layers.shape
        return img_layers


    # Clculates the coordinates of the rectangle's corners given an angle
    def __get_clock_hand_coords(self, radian_angle):

        assert (radian_angle >= 0) and (radian_angle < 2 * math.pi), 'Invalid angle.'

        size = self.img_size
        x_begin = (size + (size % 2)) / 2
        y_begin = (size + (size % 2)) / 2
        x_end = x_begin + math.cos(radian_angle) * size / 2
        y_end = y_begin + math.sin(radian_angle) * size / 2

        return([(x_begin, y_begin), (x_end, y_end)])


    def __random_angle(self):
        return random.random() * 2 * math.pi

