import numpy as np

def load_sampler(config):

    chosen_sampling = config['SAMPLE']['type']

    return {
        'toroidal': toroidal_sampler,
    }.get(chosen_sampling)


# =================================================
# SAMPLERS

def toroidal_sampler(batch_size, latent_dim):

    assert latent_dim % 2 == 0, 'latent dimension is odd.'

    z_sample = np.random.normal(size=(batch_size, latent_dim))
    l2 = np.sqrt(z_sample[:, 0::2] ** 2 + z_sample[:, 1::2] ** 2)
    l22 = np.repeat(l2, 2, axis=1)
    z_sample /= l22
    return z_sample