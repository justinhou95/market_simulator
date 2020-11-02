import gpflow
import base
import numpy as np
import tensorflow as tf
from gpsig import signature_algs

class SignatureKernel(gpflow.kernels.Kernel):

    def __init__(self, num_levels = 0, num_lags = 0, base_kernel = None, active_dims=None, order=1, difference=True, name=None):
        
        super().__init__(active_dims=active_dims)
        
        self.num_lags = num_lags
        self.num_levels = num_levels
        self.order = num_levels if (order <= 0 or order >= num_levels) else order
        self.difference = difference
        if base_kernel is None:
            self.base_kernel = gpflow.kernels.Linear()
        else:
            self.base_kernel = base_kernel
                
 
    def _K_seq(self, X, X2 = None):
        
        if self.num_lags > 0:
            X = self._lags(self.num_lags, X, weighted = True)
            if X2 is not None:
                X2 = self._lags(self.num_lags, X2, weighted = True)
                
        num_examples, len_examples, num_features = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
        
          
        if X2 is None:
            M = self.base_kernel.K(tf.reshape(X,[-1,num_features]))
            M = tf.reshape(M,shape = [num_examples, len_examples, num_examples, len_examples])
        else:
            num_examples2, len_examples2 = tf.shape(X2)[0], tf.shape(X2)[1]
            M = self.base_kernel.K(tf.reshape(X,[-1,num_features]),tf.reshape(X2,[-1,num_features]))
            M = tf.reshape(M,shape = [num_examples, len_examples, num_examples2, len_examples2])  
            
        if self.order == 1:
            K_lvls = signature_algs.signature_kern_first_order(M, self.num_levels, difference=self.difference)
        else:
            K_lvls = signature_algs.signature_kern_higher_order(M, self.num_levels, order=self.order, difference=self.difference)

        return K_lvls
            
    def K(self, X, X2 = None):
        if X2 is None:
            K_lvls = self._K_seq(X)
        else:
            K_lvls = self._K_seq(X, X2)      
        return K_lvls
    
    def _K_seq_diag(self, X):
        num_examples, len_examples, num_features = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
        M = np.array([self.base_kernel.K(x.reshape([-1,num_features])) for x in X])
        if self.order == 1:
            K_lvls_diag = signature_algs.signature_kern_first_order(M, self.num_levels, difference=self.difference)
        else:
            K_lvls_diag = signature_algs.signature_kern_higher_order(M, self.num_levels, order=self.order, difference=self.difference)
        return K_lvls_diag
    
    def K_diag(self, X):
        K_lvls = self._K_seq_diag(X)      
        return K_lvls
    
    
    def _lags(self, num_lags, X, weighted = True):
        num_examples, len_examples, num_features = X.shape[0], X.shape[1], X.shape[2]
        X = X[:,:,None,:]
        X_lag = X
        for i in range(1,num_lags+1):
            Xl = tf.concat([X[:,i:,:,:], tf.tile(X[:,-1:,:,:],[1,i,1,1])], axis=1)
            X_lag = tf.concat([X_lag,Xl],axis = 2)

        gamma = 1. / np.asarray(range(1, num_lags+2))
        gamma /= np.sum(gamma)                   
        gamma = gpflow.Parameter(gamma, dtype = 'float64')

        X_lag *= gamma[None, None, :, None]

        return tf.reshape(X_lag,shape = [num_examples,len_examples,-1])
    
    
    
    

    
    