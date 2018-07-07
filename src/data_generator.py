import logging
from sampler import load_sampler
from general_image_generator import load_image_generator

class DataGenerator():
    '''
        This class is responsible for pre-generating data for learning
        and serving that to other classes in the form of: train, dev, test sets.
    '''

    def __init__(self, config):
        self.logger = logging.getLogger('main')
        self.logger.info('DataGenerator initialised.')
        self.config = config
        self.value_sampler = load_sampler(config)
        self.image_generator = load_image_generator(config)
        self.__initialise_data_sets()

    def __initialise_data_sets(self):
        train_n = int(self.config['LEARNING']['train_n'])
        dev_n   = int(self.config['LEARNING']['dev_n'])
        test_n  = int(self.config['LEARNING']['test_n'])
        self.data = {
            'train' : self.image_generator.sample(train_n),
            'dev'   : self.image_generator.sample(dev_n),
            'test'  : self.image_generator.sample(test_n)
        }