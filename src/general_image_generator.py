import abc
import logging
from sampler import load_sampler

class ImageGenerator(abc.ABC):
    '''
        This class generates all pictures.
    '''
    def __init__(self, config):
        self.target_size = int(config['DATA']['target_size'])
        self.n_layers   = int(config['DATA']['n_layers'])
        self.antialias_factor = int(config['DATA']['antialias_factor'])
        self.logger = logging.getLogger('main')
        

    @abc.abstractmethod
    def sample(self, n):
        '''
            Generates N images in the format requested in the ini file
        '''
        raise NotImplmentedError



from image_generators import *
def load_image_generator(config):

    chosen_generator = config['DATA']['class']

    return {
        'clock': clock.Clock(config),
    }.get(chosen_generator)

