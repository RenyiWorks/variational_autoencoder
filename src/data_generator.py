import logging
from sampler import load_sampler
from general_image_generator import load_image_generator
import numpy as np

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
        self.batch_index = 0
        self.__initialise_data_sets()

    def __initialise_data_sets(self):
        train_n = int(self.config['DATA']['train_n'])
        dev_n   = int(self.config['DATA']['dev_n'])
        test_n  = int(self.config['DATA']['test_n'])
        self.data = {
            'train' : self.image_generator.sample(train_n),
            'dev'   : self.image_generator.sample(dev_n),
            'test'  : self.image_generator.sample(test_n)
        }

    def __shuffle(self):
        np.random.shuffle(self.data['train'])

    def next_batch(self, batch_size):
        last_index = self.data['train'].shape[0]
        start = self.batch_index
        end = min(start + batch_size, last_index)

        if start == 0: self.__shuffle()
        self.batch_index = 0 if end == last_index else self.batch_index + batch_size

        return self.data['train'][start:end]

    def get_batches(self, batch_size):
        batches = []
        while(1==1):
            batches.append(self.next_batch(batch_size))
            if self.batch_index == 0: break
        return batches