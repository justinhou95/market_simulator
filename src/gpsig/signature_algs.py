import numpy as np
import tensorflow as tf

def signature_kern_first_order(M, num_levels, difference = True):
    num_examples,num_examples2  = tf.shape(M)[0],tf.shape(M)[2]
    if difference == True:
        M = M[:, 1:, ..., 1:] + M[:, :-1, ..., :-1] - M[:, :-1, ..., 1:] - M[:, 1:, ..., :-1]
    K = [tf.ones((num_examples, num_examples2))]
    K.append(tf.math.reduce_sum(M, axis=(1, -1)))
    R = M
    for i in range(2, num_levels+1):
        R = M * tf.cumsum(tf.cumsum(np.sum(R,axis = (0,1)), exclusive=True, axis=1), exclusive=True, axis=-1)
        K.append(np.sum(R, axis=(1, -1)))
    K = np.array(K)
    K_1order = np.sum(K,axis = 0)
    return K_1order

def signature_kern_higher_order(M, num_levels, order=2, difference = True):
    num_examples,num_examples2  = tf.shape(M)[0],tf.shape(M)[2]
    if difference == True:
        M = M[:, 1:, ..., 1:] + M[:, :-1, ..., :-1] - M[:, :-1, ..., 1:] - M[:, 1:, ..., :-1]
    K = [tf.ones((num_examples, num_examples2))]
    K.append(tf.math.reduce_sum(M, axis=(1, -1)))
    R = np.empty(shape = (1,1), dtype=tf.Tensor)
    R[0,0] = M
    for i in range(2, num_levels+1):
        d = min(i, num_levels)
        R_next = np.empty(shape = (d,d), dtype=tf.Tensor)
        R_next[0,0] = M * tf.cumsum(tf.cumsum(np.sum(R,axis = (0,1)), exclusive=True, axis=1), exclusive=True, axis=-1)
        for j in range(2, d+1):
            R_next[0, j-1] = 1 / j * M * tf.cumsum(np.sum(R[:,j-2],axis=0), exclusive=True, axis=1)
            R_next[j-1, 0] = 1 / j * M * tf.cumsum(np.sum(R[j-2,:],axis=0), exclusive=True, axis=-1)    
            for k in range(2, d+1):
                R_next[j-1, k-1] = 1 / (j*k) * M * R[j-2, k-2]
        K.append(np.sum(np.sum(R_next, axis=(0,1)), axis=(1, -1)))
        R = R_next
    K = np.array(K)
    K_highorder = np.sum(K,axis = 0)
    K_highorder
    return K_highorder
