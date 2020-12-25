
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


def sig_stream2(path,depth_max):          # growing signature with only discrete approximation
    path = torch.Tensor(path[None,:,:])
    batch, length, channels = path.shape
    length = length-1
    index2word = index_to_word(channels, depth_max)
    dim_sig = len(index2word)+1
    sig_path_split = [signatory.signature(path[:,i:i+2,:], 1) for i in range(length)] 
    sig_path = sig_path_split[0]
    dim_0 = signatory.signature_channels(channels,depth_max) - signatory.signature_channels(channels,1)
    helper0 = torch.zeros(size = [1,dim_0])
    sig_path_aug = torch.cat([sig_path,helper0],axis = -1)
    sig_path_stream = [sig_path_aug[:,None,:]]
    for i in range(len(sig_path_split)-1):
        depth_now = min(i + 2,depth_max)
        depth_pre = min(i + 1,depth_max)
        dim_0 = signatory.signature_channels(channels,depth_max) - signatory.signature_channels(channels,depth_now)
        dim_1 = signatory.signature_channels(channels,depth_now) - signatory.signature_channels(channels,depth_pre)
        dim_2 = signatory.signature_channels(channels,depth_now) - signatory.signature_channels(channels,1)
        helper0 = torch.zeros(size = [1,dim_0])
        helper1 = torch.zeros(size = [1,dim_1])
        helper2 = torch.zeros(size = [1,dim_2])

        sig1 = torch.cat([sig_path,helper1],axis = -1)
        sig2 = torch.cat([sig_path_split[i+1],helper2],axis = -1)
        sig_path = signatory.signature_combine(sig1, sig2, channels, depth_now)
        sig_path_aug = torch.cat([sig_path,helper0],axis = -1)
        sig_path_stream.append(sig_path_aug[:,None,:])

    sig_path_stream = torch.cat(sig_path_stream,axis = 1)
    sig_path_stream = torch.cat([torch.zeros([batch,1,dim_sig-1]),sig_path_stream],axis = 1)
    sig_path_stream = torch.cat([torch.ones([batch,length+1,1]),sig_path_stream],axis = 2)
    return sig_path_stream

def SDEfromSig(BMpath,initial,depth,B):
    channels = BMpath.shape[-1]
    index2word = index_to_word(channels, depth)
    sig_path_stream = sig_stream2(BMpath,depth).numpy()
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

