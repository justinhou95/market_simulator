import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

tf.keras.backend.set_floatx('float32')

def Encoder(data_dim, cond_dim, latent_dim, n_hidden):
    inputs_data = tfk.Input(shape=(data_dim,))
    inputs_cond = tfk.Input(shape=(cond_dim,))
    x = tfkl.concatenate([inputs_data,inputs_cond],axis = 1)
    x = tfkl.Dense(units=n_hidden)(x)
    x = tfkl.LeakyReLU(alpha=0.3)(x)
    mn = tfkl.Dense(units=latent_dim)(x)
    mn = tfkl.LeakyReLU(alpha=0.3)(mn)
    sd = tfkl.Dense(units=latent_dim)(x)
    sd = tfkl.LeakyReLU(alpha=0.3)(sd)
    return tfk.models.Model([inputs_data, inputs_cond], [mn, sd])

def Decoder(data_dim, cond_dim, latent_dim, n_hidden):
    inputs_z = tfk.Input(shape=(latent_dim,))
    inputs_label = tfk.Input(shape=(cond_dim,))
    x = tfkl.concatenate([inputs_z, inputs_label], axis=1)
    x = tfkl.Dense(units = n_hidden)(x)
    x = tfkl.LeakyReLU(alpha=0.3)(x)
    x = tfkl.Dense(units = data_dim, activation = 'sigmoid')(x)
    return tfk.models.Model([inputs_z, inputs_label], x)

class EncoderNormal(tf.keras.Model):
    def __init__(self,data_dim, cond_dim, latent_dim, n_hidden):
        super(EncoderNormal, self).__init__()
        self.net = Encoder(data_dim, cond_dim, latent_dim, n_hidden)
        self.latent_dim = latent_dim
    def call(self, x):
        inputs, peudo_inputs = x
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim),scale_diag=tf.ones(self.latent_dim))
        mn, std = self.net(inputs)
        self.posterior = tfd.MultivariateNormalDiag(loc=mn,scale_diag=tf.exp(0.5 * std))
        return [self.prior, self.posterior]
    def call_prior(self,peudo_inputs):
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim),scale_diag=tf.ones(self.latent_dim))
        return self.prior
    
class EncoderStudent(tf.keras.Model):
    def __init__(self,data_dim, cond_dim, latent_dim = 4, n_hidden = 50):
        super(EncoderStudent, self).__init__()
        self.net = Encoder(data_dim, cond_dim, latent_dim, n_hidden)
        self.latent_dim = latent_dim
    def call(self, x):
        inputs, peudo_inputs = x
        self.df = tf.exp(0.5 * peudo_inputs[1])[0] * tf.ones(self.latent_dim)
        self.prior = tfd.Independent(tfp.distributions.StudentT(loc=tf.zeros(self.latent_dim),scale=tf.ones(self.latent_dim)\
                                                           ,df=self.df),reinterpreted_batch_ndims = 1)
        # real inputs
        mn, std = self.net(inputs)
        self.posterior = tfd.Independent(tfp.distributions.StudentT(loc = mn , scale= tf.exp(0.5 * std), df = self.df ),
                                            reinterpreted_batch_ndims = 1)
        return [self.prior, self.posterior]
    def call_prior(self,peudo_inputs):
        self.df = tf.exp(0.5 * peudo_inputs[1])[0] * tf.ones(self.latent_dim)
        self.prior = tfd.Independent(tfp.distributions.StudentT(loc=tf.zeros(self.latent_dim),scale=tf.ones(self.latent_dim)\
                                                           ,df=self.df),reinterpreted_batch_ndims = 1)
        return self.prior
    
class EncoderStudentPseudo(tf.keras.Model):
    def __init__(self,data_dim, cond_dim, latent_dim = 4, n_hidden = 50):
        super(EncoderStudentPseudo, self).__init__()
        self.net = Encoder(data_dim, cond_dim, latent_dim, n_hidden)
        self.latent_dim = latent_dim
    def call(self, x):
        inputs, peudo_inputs = x
        # pseudo inputs
        self.df = tf.exp(0.5 * self.net(peudo_inputs)[1])[0]
        self.prior = tfd.Independent(tfp.distributions.StudentT(loc=tf.zeros(self.latent_dim),scale=tf.ones(self.latent_dim)\
                                                           ,df=self.df),reinterpreted_batch_ndims = 1)
        # real inputs
        mn, std = self.net(inputs)
        self.posterior = tfd.Independent(tfp.distributions.StudentT(loc = mn , scale= tf.exp(0.5 * std), df = self.df ),
                                            reinterpreted_batch_ndims = 1)
        return [self.prior, self.posterior]
    def call_prior(self,peudo_inputs):
        self.df = tf.exp(0.5 * self.net(peudo_inputs)[0])[0]
        self.prior = tfd.Independent(tfp.distributions.StudentT(loc=tf.zeros(self.latent_dim),scale=tf.ones(self.latent_dim)\
                                                           ,df=self.df),reinterpreted_batch_ndims = 1)
        return self.prior    
    
    
    

class CVAE(tf.keras.Model):
    def __init__(self, data_dim, cond_dim, latent_dim = 4, n_hidden = 50, weight = 0.02,\
                 decoder = Decoder, encoder = EncoderStudent):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.n_hidden = n_hidden
        self.weight = weight
        self.df_input = [tf.Variable([[0.,0.]], trainable=True),tf.Variable([[0.]], trainable=True)]
        self.decoder = decoder(data_dim, cond_dim, latent_dim, n_hidden)
        self.encoder = encoder(data_dim, cond_dim, latent_dim, n_hidden)
        self.K = 1
    
    def call(self, x_input):
        # main vae        
        inputs_data, inputs_cond = x_input
        [self.prior, self.posterior] = self.encoder([x_input,self.df_input])
        z = self.posterior.sample(self.K)
        z = tf.reshape(z,[-1,self.latent_dim])
        inputs_cond_expand = tf.tile(inputs_cond, [self.K, 1]) # (K*Batch,d)
        x = self.decoder([z, inputs_cond_expand])
        # compute loss
        inputs_data_expand = tf.tile(inputs_data, [self.K, 1]) # (K*Batch,d)
        decoded_loss = tf.reduce_sum(tf.square(x - inputs_data_expand), axis = -1)
        latent_loss = self.posterior._log_prob(z) - self.prior._log_prob(z)
        loss = (1 - self.weight) * decoded_loss + self.weight * latent_loss 
        self.add_loss(tf.math.reduce_mean(loss))
        return x
    
class CVAE_circle(CVAE):    
    def generate(self, conditions):
        self.prior = self.encoder.call_prior(self.df_input)
        sample_dim =  conditions.shape[0]
        randoms = self.prior.sample(sample_dim)
        outputs = self.decoder([tf.constant(randoms), tf.constant(conditions)])
        return outputs 
    
    def visualize(self):
        n_samples = 1000
        outputs0 = self.generate(np.zeros(shape=(n_samples, 1)))
        outputs1 = self.generate(np.zeros(shape=(n_samples, 1))+1)
        plt.scatter(*outputs0.numpy().T, s=2)
        plt.scatter(*outputs1.numpy().T, s=2)
        plt.show()
    


#         latent_loss = -0.5 * tf.reduce_sum(1. + sd - tf.square(mn) - tf.exp(sd), axis = -1)
#         post_logprob = tf.reduce_sum(-0.5 * tf.math.square((z-mn)/std) - \
#                                      tf.constant(0.5 * np.log(2. * np.pi),dtype = np.float32) - tf.math.log(std),axis = -1)
#         pri_logprob = tf.reduce_sum(-0.5 * tf.math.square(z) - tf.constant(0.5 * np.log(2. * np.pi),dtype = np.float32),axis = -1)
#         latent_loss = post_logprob - pri_logprob