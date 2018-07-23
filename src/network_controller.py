import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NetworkController():

    def __init__(self, config):
        self.logger = logging.getLogger('main')
        self.img_size = int(config['DATA']['target_size'])
        self.n_layers   = int(config['DATA']['n_layers'])
        self.n_latent = int(config['NETWORK']['n_latent'])
        self.kernel_size = int(config['NETWORK']['kernel_size'])

        self.encoder_filters = self.__to_list(config['NETWORK']['encoder_filters'], int)
        self.encoder_strides = self.__to_list(config['NETWORK']['encoder_strides'], int)
        self.decoder_filters = self.__to_list(config['NETWORK']['decoder_filters'], int)
        self.decoder_strides = self.__to_list(config['NETWORK']['decoder_strides'], int)

        self.encoder_activation = self.__str2activation(config['NETWORK']['encoder_activation'])
        self.decoder_activation = self.__str2activation(config['NETWORK']['decoder_activation'])

        self.decoder_input_size = int(config['NETWORK']['decoder_input_size'])
        self.decoder_reconv_size = int(config['NETWORK']['decoder_reconv_size'])

        self.n_epochs = int(config['LEARNING']['n_epochs'])
        self.batch_size = int(config['LEARNING']['batch_size'])
        self.log_step = int(config['LEARNING']['log_step'])
        self.learning_rate = float(config['LEARNING']['learning_rate'])

        self.tensorboard_log = config['PATH']['tensorboard_file']

        self.build()

    def train(self, data_generator):
        # Variables
        n = self.img_size
        l = self.n_layers
        counter = 0

        # for tensorboard
        tf.summary.scalar('loss', self.operations['loss'])
        train_writer = tf.summary.FileWriter(self.tensorboard_log, self.sess.graph)

        for i in range(self.n_epochs):
            
            batches = data_generator.get_batches(self.batch_size)
            # print()
            # print(batches)
            # print()
            # print()

            for batch in batches:
                counter += 1
                # print(batch)
                merge = tf.summary.merge_all()
                summary, _ = self.sess.run([merge, self.optimizer], feed_dict = {self.X: batch, self.Y: batch})
                train_writer.add_summary(summary, counter)

                
            if not i % self.log_step:
                loss, prediction, reconstruction_loss, latent_loss, mean, sd = self.sess.run(
                    list(self.operations.values()),
                    feed_dict = {self.X: batches[0], self.Y: batches[0]}
                )
                # plt.imshow(np.reshape(batches[0][0], [n,n,l]))
                # plt.show()
                # plt.imshow(prediction[0])
                # plt.show()
                print(i, loss, np.mean(reconstruction_loss), np.mean(latent_loss))

    def build(self):

        # Variables
        n = self.img_size
        l = self.n_layers

        # tf init
        tf.reset_default_graph()        
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, n, n, l], name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, n, n, l], name='Y')
        self.Y_flat = tf.reshape(self.Y, shape=[-1, n*n*l])

        z, mean, sd = self.__encoder(self.X)
        prediction = self.__decoder(z)
        prediction_flat = tf.reshape(prediction, [-1, n*n*l])

        reconstruction_loss = tf.reduce_sum(tf.squared_difference(prediction_flat, self.Y_flat), 1)
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mean) - tf.exp(2.0 * sd), 1)
        loss = tf.reduce_mean(reconstruction_loss + latent_loss)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.operations = {
            'loss' : loss,
            'prediction' : prediction,
            'reconstruction_loss' : reconstruction_loss,
            'latent_loss' : latent_loss,
            'mean' : mean,
            'sd' : sd
        }

    def __encoder(self, inputs):

        # Variables
        n = self.img_size
        l = self.n_layers
        filters = self.encoder_filters
        strides = self.encoder_strides
        activation = self.encoder_activation

        with tf.variable_scope("encoder", reuse=None):

            # Make sure we have the right shape
            x = tf.reshape(inputs, shape=[-1, n, n, l])

            # Conolutional layers
            for i, (n_filters, n_strides) in enumerate(zip(filters, strides)):
                x = tf.layers.conv2d(
                    inputs = x,
                    filters = n_filters,
                    kernel_size = self.kernel_size,
                    strides = n_strides,
                    padding = 'same',
                    activation = activation,
                    name = 'en_conv_' + str(i)
                )

            x = tf.contrib.layers.flatten(x)

            mean    = tf.layers.dense(x, units=self.n_latent, name='mean')
            sd      = tf.layers.dense(x, units=self.n_latent, name='sd') * 0.5
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))

            z = tf.add(mean, tf.multiply(epsilon, tf.exp(sd)), name='latent')
            
        return z, mean, sd


    def __decoder(self, sampled_z):

        # Variables
        n = self.img_size
        l = self.n_layers
        k = self.decoder_reconv_size
        filters = self.decoder_filters
        strides = self.decoder_strides
        activation = self.decoder_activation
        in_size = self.decoder_input_size

        # Make sure we reconstruct the same sized image
        assert np.prod(strides)*k==n, 'Reconstruction sizes do not match.'

        with tf.variable_scope("decoder", reuse=None):

            # Flatten from z
            x = tf.layers.dense(sampled_z, units=in_size, activation=activation)
            x = tf.layers.dense(x, units=l*k*k, activation=activation)
            x = tf.reshape(x, [-1, k, k, l])

            # Conolutional layers
            for i, (n_filters, n_strides) in enumerate(zip(filters, strides)):
                x = tf.layers.conv2d_transpose(
                    inputs = x,
                    filters = n_filters,
                    kernel_size = self.kernel_size,
                    strides = n_strides,
                    padding = 'same',
                    activation = activation,
                    name = 'de_conv_' + str(i)
                )
            
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=n*n*l, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[-1, n, n, l])

        return img



    def __str2activation(self, s):
        return {
            'relu' : tf.nn.relu,
            'leaky_relu' : tf.nn.leaky_relu,
            'sigmoid' : tf.sigmoid,
            'tanh' : tf.tanh
        }.get(s)

    def __to_list(self, str, out_type):
        return [out_type(x) for x in str.split(',')]