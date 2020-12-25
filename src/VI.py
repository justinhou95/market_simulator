import numpy as np
import matplotlib.pyplot as plt
from sklearn import utils
from tqdm.auto import tqdm
import tensorflow as tf
import importlib
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64')


class VAE(object):
    """Variational Auto Encoder (VAE)."""

    def __init__(self, n_latent, n_hidden, alpha):
        
        self.alpha = alpha
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        
    def build(self, input_dim):
        self.input_dim = input_dim
        class Sampling(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
        # Encoder 
        inputs_data = keras.Input(shape=(self.input_dim,))
        x = inputs_data
        x = layers.Dense(units=self.n_hidden)(x)
        x = layers.LeakyReLU(alpha=0.3)(x)
        mn = layers.Dense(units=self.n_latent)(x)
        mn = layers.LeakyReLU(alpha=0.3)(mn)
        sd = layers.Dense(units=self.n_latent)(x)
        sd = layers.LeakyReLU(alpha=0.3)(sd)
        z = Sampling()([mn, sd])
        self.encoder = keras.models.Model(inputs_data, [mn, sd, z])
  
        # Decoder 
        inputs_z = keras.Input(shape=(self.n_latent,))
        x = inputs_z
        x = layers.Dense(units = self.n_hidden)(x)
        x = layers.LeakyReLU(alpha=0.3)(x)
        x = layers.Dense(units = self.input_dim, activation = 'sigmoid')(x)
        self.decoder = keras.models.Model(inputs_z, x)
        
        # VAE
        inputs_data = keras.Input(shape=(self.input_dim,))
        mn, sd, z = self.encoder(inputs_data)
        x = self.decoder(z)
        reconstruction = x
        decoded_loss = tf.reduce_sum(tf.square(reconstruction - inputs_data), axis = -1, keepdims = True)
        latent_loss = -0.5 * tf.reduce_sum(1. + sd - tf.square(mn) - tf.exp(sd), axis = -1, keepdims = True)
        loss = (1 - self.alpha) * decoded_loss + self.alpha * latent_loss
        self.cvae = keras.models.Model(inputs_data, loss)
        
     
        
    def train(self, data,n_epochs=10000):
        self.cvae.compile(optimizer='adam', loss='mae')
        self.cvae.fit(x = data, y = np.zeros(shape= (data.shape[0],1)), epochs=n_epochs,\
                 batch_size=data.shape[0], verbose = True)
        self.cvae.evaluate(x = data, y = np.zeros(shape= (data.shape[0],1)))


    def generate(self):
        randoms = np.random.normal(0, 1, size=(1, self.n_latent))
        outputs = self.decoder(tf.constant(randoms))
#         plt.scatter(*outputs.numpy().T, s=2)

        return outputs


