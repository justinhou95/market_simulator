import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

tf.keras.backend.set_floatx('float32')

def zero_error(y_true, y_pred):
    return tf.constant(0., dtype='float32')

# Structure Encoder ---> Sampler ---> Decoder 

def Encoder0(data_dim, cond_dim, latent_dim, hidden_dim):    
    inputs_data0 = tfk.Input(shape=(data_dim,))
    inputs_cond0 = tfk.Input(shape=(cond_dim,))
    x0 = tfkl.concatenate([inputs_data0,inputs_cond0],axis = 1)
    x0 = tfkl.Dense(units=hidden_dim)(x0)
    x0 = tfkl.LeakyReLU(alpha=0.3)(x0)
    mn0 = tfkl.Dense(units=latent_dim)(x0)
    mn0 = tfkl.LeakyReLU(alpha=0.3)(mn0)
    sd0 = tfkl.Dense(units=latent_dim)(x0)
    sd0 = tfkl.LeakyReLU(alpha=0.3)(sd0)
    sd0 = tf.exp(0.5 * sd0)
    model0 = tfk.models.Model([inputs_data0, inputs_cond0], [mn0,sd0])
    return model0

class Encoder(tf.keras.Model):
    def __init__(self, data_dim, cond_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.coeff_prior_dim = 0
        self.coeff_posterior_dim = 2
        self.encoder0 = Encoder0(data_dim, cond_dim, latent_dim, hidden_dim)
    def call(self, inputs):
        mn, sd = self.encoder0(inputs)
        return [mn, sd]
    
class Encoder_Pseudo(tf.keras.Model):
    def __init__(self, data_dim, cond_dim, latent_dim, hidden_dim):
        super(Encoder_Pseudo, self).__init__()
        self.coeff_prior_dim = 1
        self.coeff_posterior_dim = 2
        self.encoder0 = Encoder0(data_dim, cond_dim, latent_dim, hidden_dim)
        init = tf.zeros_initializer()
        self.pseudo_data = tf.Variable(initial_value=init(shape=(1,data_dim,), dtype="float32"), trainable=True)
        self.pseudo_cond = tf.Variable(initial_value=init(shape=(1,cond_dim,), dtype="float32"), trainable=True)
        self.pseudo_input = [self.pseudo_data, self.pseudo_cond]
    def call(self, inputs):
        mn,sd = self.encoder0(inputs)
        _,pseudo_output = self.encoder0(self.pseudo_input)
        pseudo_output = pseudo_output[0,0]
        return [pseudo_output, mn, sd]     

def Decoder(data_dim, cond_dim, latent_dim, hidden_dim):
    inputs_z = tfk.Input(shape=(latent_dim,))
    inputs_label = tfk.Input(shape=(cond_dim,))
    x = tfkl.concatenate([inputs_z, inputs_label], axis=1)
    x = tfkl.Dense(units = hidden_dim)(x)
    x = tfkl.LeakyReLU(alpha=0.3)(x)
    x = tfkl.Dense(units = data_dim, activation = 'sigmoid')(x)
    return tfk.models.Model([inputs_z, inputs_label], x)

class Sampler:
    def __init__(self,latent_dim):
        self.latent_dim = latent_dim
    def posterior(self,coeff_all):
        return self.distribution(coeff_all)
    def prior(self,coeff_prior):
        coeff_all = coeff_prior + self.coeff_post
        return self.distribution(coeff_all)
    
class NormalSampler(Sampler):
    def __init__(self,latent_dim):
        super(NormalSampler, self).__init__(latent_dim)
        self.loc = tf.zeros(self.latent_dim)
        self.scale_diag = tf.ones(self.latent_dim)
        self.coeff_post  = [self.loc, self.scale_diag]
    def distribution(self, coeff_all):
        loc, scale_diag = coeff_all
        return tfd.MultivariateNormalDiag(loc=loc,scale_diag=scale_diag)
         
class StudentSampler(Sampler):
    def __init__(self,latent_dim):
        super(StudentSampler, self).__init__(latent_dim)  
        self.loc = tf.zeros(self.latent_dim)
        self.scale = tf.ones(self.latent_dim)
        self.coeff_post = [self.loc, self.scale]
    def distribution(self, coeff_all):
        df, loc, scale  = coeff_all
        return tfd.Independent(tfp.distributions.StudentT(loc=loc,scale=scale,df=df),\
                               reinterpreted_batch_ndims = 1) 

class EncoderSampler(tf.keras.Model):
    def __init__(self,encoder,sampler):
        super(EncoderSampler, self).__init__()
        self.encoder = encoder
        self.sampler = sampler
    def call(self, inputs):
        coeff_all = self.encoder(inputs)
        coeff_prior = coeff_all[:self.encoder.coeff_prior_dim]
        prior = self.sampler.prior(coeff_prior)
        posterior = self.sampler.posterior(coeff_all)
        return [prior, posterior]
    

class CVAE(tf.keras.Model):
    def __init__(self, data_dim, cond_dim, latent_dim, hidden_dim, weight, decoder, encodersampler):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.weight = weight
        self.encodersampler = encodersampler
        self.decoder = decoder
        self.K = 1
    
    def call(self, x_input):
        # encoder and sampler    
        inputs_data, inputs_cond = x_input
        prior, posterior = self.encodersampler(x_input)
        z = posterior.sample(self.K)     # augmentation sampling
        z = tf.reshape(z,[-1,self.latent_dim])
        # decoder
        inputs_cond_expand = tf.tile(inputs_cond, [self.K, 1]) # (K*Batch,d)
        x_output = self.decoder([z, inputs_cond_expand])
        # compute loss
        inputs_data_expand = tf.tile(inputs_data, [self.K, 1]) # (K*Batch,d)
        decoded_loss = tf.reduce_sum(tf.square(x_output - inputs_data_expand), axis = -1)
        latent_loss = posterior._log_prob(z) - prior._log_prob(z)
        loss = (1 - self.weight) * decoded_loss + self.weight * latent_loss 
        self.add_loss(tf.math.reduce_mean(loss))
        return x_output
    
    
    
    
    
    
class CVAE_circle(CVAE):    
    def generate(self, condition):
        sample_dim =  condition.shape[0]
        x_inputs = [np.zeros([sample_dim,self.data_dim]),np.zeros([sample_dim,self.cond_dim])]
        prior,_ = self.encodersampler(x_inputs)
        randoms = prior.sample(sample_dim)
        outputs = self.decoder([tf.constant(randoms), tf.constant(condition)])
        return outputs 
    
    def visualize(self):
        n_samples = 1000
        for i in range(2):
            condition = np.zeros(shape=(n_samples, 1)) + i
            outputs = self.generate(condition)
            plt.scatter(*outputs.numpy().T, s=2)
        plt.show()
        
        
        
class CVAE_bergomi(CVAE):    
    def generate(self, condition):
        sample_dim =  condition.shape[0]
        x_inputs = [np.zeros([sample_dim,self.data_dim]),np.zeros([sample_dim,self.cond_dim])]
        prior,_ = self.encodersampler(x_inputs)
        randoms = prior.sample(sample_dim)
        outputs = self.decoder([tf.constant(randoms), tf.constant(condition)])
        return outputs 


#         latent_loss = -0.5 * tf.reduce_sum(1. + sd - tf.square(mn) - tf.exp(sd), axis = -1)
#         post_logprob = tf.reduce_sum(-0.5 * tf.math.square((z-mn)/std) - \
#                                      tf.constant(0.5 * np.log(2. * np.pi),dtype = np.float32) - tf.math.log(std),axis = -1)
#         pri_logprob = tf.reduce_sum(-0.5 * tf.math.square(z) - tf.constant(0.5 * np.log(2. * np.pi),dtype = np.float32),axis = -1)
#         latent_loss = post_logprob - pri_logprob