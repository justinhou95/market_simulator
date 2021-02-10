import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import numpy as np
import scipy as sp
from scipy.linalg import expm
from matplotlib import pyplot as plt
import torch
torch.set_default_dtype(torch.float64)
import signatory


def index_to_word(channels, depth):
    index2word = signatory.all_words(channels, depth)
    return index2word
def word_to_index(channels, depth):
    word2index = {}
    index2word = signatory.all_words(channels, depth)
    for i,word in enumerate(index2word):
        word2index.update({word: i})
    return word2index

def B_aug(B,depth):
    dimension = B.shape[-1]
    Id = np.eye(dimension)
    BB = [Id]
    index2word = index_to_word(dimension, depth)
    for word in index2word:    # dp can also be used here
        M = np.eye(dimension)
        for i in word:
            M =  B[i,:,:] @ M
        BB.append(M)
    return BB


# def sig_stream2(path,depth_max):          # growing signature with only discrete approximation
#     path = torch.Tensor(path[None,:,:])
#     batch, length, channels = path.shape
#     length = length-1
#     index2word = index_to_word(channels, depth_max)
#     dim_sig = len(index2word)+1
#     sig_path_split = [signatory.signature(path[:,i:i+2,:], 1) for i in range(length)] 
#     sig_path = sig_path_split[0]
#     dim_0 = signatory.signature_channels(channels,depth_max) - signatory.signature_channels(channels,1)
#     helper0 = torch.zeros(size = [1,dim_0])
#     sig_path_aug = torch.cat([sig_path,helper0],axis = -1)
#     sig_path_stream = [sig_path_aug[:,None,:]]
#     for i in range(len(sig_path_split)-1):
#         depth_now = min(i + 2,depth_max)
#         depth_pre = min(i + 1,depth_max)
#         dim_0 = signatory.signature_channels(channels,depth_max) - signatory.signature_channels(channels,depth_now)
#         dim_1 = signatory.signature_channels(channels,depth_now) - signatory.signature_channels(channels,depth_pre)
#         dim_2 = signatory.signature_channels(channels,depth_now) - signatory.signature_channels(channels,1)
#         helper0 = torch.zeros(size = [1,dim_0])
#         helper1 = torch.zeros(size = [1,dim_1])
#         helper2 = torch.zeros(size = [1,dim_2])

#         sig1 = torch.cat([sig_path,helper1],axis = -1)
#         sig2 = torch.cat([sig_path_split[i+1],helper2],axis = -1)
#         sig_path = signatory.signature_combine(sig1, sig2, channels, depth_now)
#         sig_path_aug = torch.cat([sig_path,helper0],axis = -1)
#         sig_path_stream.append(sig_path_aug[:,None,:])

#     sig_path_stream = torch.cat(sig_path_stream,axis = 1)
#     sig_path_stream = torch.cat([torch.zeros([batch,1,dim_sig-1]),sig_path_stream],axis = 1)
#     sig_path_stream = torch.cat([torch.ones([batch,length+1,1]),sig_path_stream],axis = 2)
#     return sig_path_stream

def sig_stream2(path,depth_max):
    len_sequences, num_features = path.shape
    sequence = tf.constant(path[None,:,:],dtype = 'float64')
    l_sig = GrowingSignature(depth_max)
    sig = l_sig(sequence)
    return sig
    

def SDEfromSig(BMpath,initial,depth,B):
    channels = BMpath.shape[-1]
    index2word = index_to_word(channels, depth)
    
    sig_path_stream = sig_stream2(BMpath,depth).numpy()
    
    BB = B_aug(B,depth)
    CC = np.array([M @ initial for M in BB])
    SDEpath_by_signature = np.dot(sig_path_stream,CC)
    return SDEpath_by_signature[0,:,:]

def SDEfromSig_layer(BMpath,initial,depth,B):
    channels = BMpath.shape[-1]
    index2word = index_to_word(channels, depth)
    
    sequence = tf.constant(BMpath[None,:,:],dtype = 'float64')
    l_sig = GrowingSignature(depth)
    sig = l_sig(sequence)
    sig_path_stream = sig.numpy()
    
    BB = B_aug(B,depth)
    CC = np.array([M @ initial for M in BB])
    SDEpath_by_signature = np.dot(sig_path_stream,CC)
    return SDEpath_by_signature[0,:,:]


def sig_vectorfield(channels, depth):
    index2word = index_to_word(channels, depth)
    word2index = word_to_index(channels, depth)
    dim_sig = len(index2word)+1
    V = [np.zeros([dim_sig, dim_sig]) for i in range(channels)]
    for i in range(dim_sig):
        if i > 0:
            res = (i-1)%channels
            word = index2word[i-1]
            if len(word) == 1:
                V[res][i,0] = 1
            else:
                word_reduce = tuple(list(word)[:-1])
                V[res][i,word2index[word_reduce]+1] = 1
    return V

def semi_group_sig(state,increment,dt,Vec):
    dimension = state.shape[-1]
    I = np.eye(dimension)
    a = np.tensordot(increment,Vec,axes = 1)
    V = I + a
    dX = V@state
    return dX


##########################
class GrowingSignature(tf.keras.layers.Layer):

    def __init__(self, 
                 num_levels,
                 **kwargs):
        super(GrowingSignature, self).__init__(**kwargs)
        self.num_levels = num_levels

    def build(self, input_shape):
        self.num_features = input_shape.as_list()[-1]
        
        assert len(input_shape) == 3
        index2word = signatory.all_words(self.num_features, self.num_levels)
        self.num_components = int(self.num_levels * (self.num_levels+1) / 2.)
        self.num_functionals = len(index2word)
        self.kernel = kernel_for_signature(self.num_components, self.num_features, self.num_levels, self.num_functionals)
        super(GrowingSignature, self).build(input_shape)
        
    def call(self, X, mask=None):
        num_sequences, len_sequences, _ = tf.unstack(tf.shape(X))
        seq = X[:,1:,:] - X[:,:-1,:]
        
        sig = low_rank_seq2tens(seq, self.kernel, self.num_levels, embedding_order=1, recursive_weights=False,
                                 bias=None, reverse=False, return_sequences=True, mask=None)
        sig = tf.reduce_sum(sig,axis = 2)
      
        helper = tf.zeros(shape = [num_sequences, 1, self.num_functionals], dtype = 'float64')
        sig = tf.concat([helper,sig],axis = 1)
        helper =tf.ones(shape = [num_sequences, len_sequences, 1], dtype = 'float64')
        sig = tf.concat([helper,sig],axis = -1)
        return sig
        

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.num_levels, self.num_functionals)
        else:
            return (input_shape[0], self.num_levels, self.num_functionals)


def kernel_for_signature(num_components, num_features, num_levels, num_functionals):
    index2word = signatory.all_words(num_features, num_levels)
    kernel = np.zeros(shape = [num_components, num_features, num_functionals])
    for i,word in enumerate(index2word):
        m = len(word)
        start = int(m*(m-1)/2)
        for j,axis in enumerate(word):
            kernel[start + j,axis,i] = 1
    return tf.constant(kernel, dtype = 'float64')


def low_rank_seq2tens(sequences, kernel, num_levels, embedding_order=1,\
                      recursive_weights=False, bias=None, reverse=False, return_sequences=False, mask=None):
    """
    Tensorflow implementation of the Low-rank Seq2Tens (LS2T) map
    --------------------------------------------------
    Args
    ----
    :sequences: - a tensor of sequences of shape (num_examples, len_examples, num_features)
    :kernel: - a tensor of component vectors of rank-1 weight tensors of shape (num_components, num_features, num_functionals)
    :num_levels: - an int scalar denoting the cutoff degree in the features themselves (must be consistent with the 'num_components' dimension of 'kernel')
    :embedding_order: - an int scalar denoting the cutoff degree in the algebraic embedding
    :recursive_weights: - whether the rank-1 weight twensors are contructed in a recursive way (must be consistent with the shape of 'kernel')
    :bias: - a tensor of biases of shape (num_components, num_functionals)
    :reverse: - only changes the results with 'return_sequences=True', determines whether the output sequences are constructed by moving the starting point or ending point of subsequences
    """
    
    num_sequences, len_sequences, num_features = tf.unstack(tf.shape(sequences))

    num_components = int(num_levels * (num_levels+1) / 2.) if not recursive_weights else num_levels
    
    num_functionals = tf.shape(kernel)[-1]
        
    M = tf.matmul(tf.reshape(sequences, [1, -1, num_features]), kernel)
        
    M = tf.reshape(M, [num_components, num_sequences, len_sequences, num_functionals])
    
    if bias is not None:
        M += bias[:, None, None, :]
    
    if mask is not None:
        M = tf.where(mask[None, :, :, None], M, tf.zeros_like(M))

    if embedding_order == 1:
        if recursive_weights:
            return _low_rank_seq2tens_first_order_embedding_recursive_weights(M, num_levels, reverse=reverse, return_sequences=return_sequences)
        else:
            return _low_rank_seq2tens_first_order_embedding_indep_weights(M, num_levels, reverse=reverse, return_sequences=return_sequences)
    else:
        if recursive_weights:
            return _low_rank_seq2tens_higher_order_embedding_recursive_weights(M, num_levels, embedding_order, reverse=reverse, return_sequences=return_sequences)
        else:
            return _low_rank_seq2tens_higher_order_embedding_indep_weights(M, num_levels, embedding_order, reverse=reverse, return_sequences=return_sequences)

def _low_rank_seq2tens_first_order_embedding_recursive_weights(M, num_levels, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    R = M[0]
    for m in range(1, num_levels):
        R = M[m] * tf.cumsum(R, exclusive=True, reverse=reverse, axis=1)
        
        if return_sequences:
            Y.append(tf.cumsum(R, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R, axis=1))

    return tf.stack(Y, axis=-2)
    
def _low_rank_seq2tens_first_order_embedding_indep_weights(M, num_levels, reverse=False, return_sequences=False):
    
    if return_sequences:
        Y = [tf.cumsum(M[0], reverse=reverse, axis=1)]
    else:
        Y = [tf.reduce_sum(M[0], axis=1)]

    k = 1
    for m in range(1, num_levels):
        R = M[k]
        k += 1
        for i in range(1, m+1):
            R = M[k] *  tf.cumsum(R, exclusive=True, reverse=reverse, axis=1)
            k += 1
        if return_sequences:
            Y.append(tf.cumsum(R, reverse=reverse, axis=1))
        else:
            Y.append(tf.reduce_sum(R, axis=1))
    
    return tf.stack(Y, axis=-2)
