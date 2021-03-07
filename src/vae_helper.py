import matplotlib.pyplot as plt
from esig.tosig import sigkeys
import numpy as np
from esig.tosig import logsigkeys
import re

import base
from utils import tosig
from utils.leadlag import leadlag
from rough_bergomi import rough_bergomi

import torch
import signatory 
import normalize

from sklearn.preprocessing import MinMaxScaler

import evaluation
import process_discriminator




def plot_sig(order, train_sig, test_sig, generated_sig1,different_sig = None, plot_all = False, labels = None):
    if labels is None:
        labels = ["Different data", "Train data", "Test data", "VAE normal", ]

    keys = sigkeys(2, order).split()
    factor = []
    for i in keys:
        factor.append((np.ceil((len(i)-2)/2)))
    factor = np.array(factor)

#     PROJECTIONS = [(4, 5), (3, 4), (7, 8), (9, 10)]
    PROJECTIONS = [(4, 5), (2, 4), (6, 7), (9, 2)]
    f,p=plt.subplots(1,4,figsize=(20,3))
    for k, projection in enumerate(PROJECTIONS):
        
        if different_sig is not None:
            p[k].scatter(different_sig[:, projection[0]], different_sig[:, projection[1]],\
                       label=labels[0], alpha = 0.7, color = 'tab:red')
            
        if different_sig is None or plot_all is True:
            p[k].scatter(train_sig[:, projection[0]], train_sig[:, projection[1]],\
                       label=labels[1], alpha = 0.7, color = 'tab:blue')
         
        p[k].scatter(test_sig[:, projection[0]], test_sig[:, projection[1]],\
                       label=labels[2], alpha = 0.7, color = 'tab:green')
        p[k].scatter(generated_sig1[:, projection[0]], generated_sig1[:, projection[1]],\
                         label=labels[3], alpha = 0.7, color = 'tab:orange')
        
        p[k].set_xlabel(keys[projection[0]], fontsize=10)
        p[k].set_ylabel(keys[projection[1]], fontsize=10)
        p[k].legend()
        p[k].grid()
    plt.suptitle('Signature')
    plt.show()
    
def plot_logsig(order, train_logsig, test_logsig, generated_logsig1, different_logsig = None, plot_all = False, labels = None):
    if labels is None:
        labels = ["Different data", "Train data", "Test data", "VAE normal", ]
    logkeys = logsigkeys(2, order).split()
    logfactor = []
    for i in logkeys:
        logfactor.append((len(re.sub("\D", "", i))))
    logfactor = np.array(logfactor)
    PROJECTIONS = [(4, 5), (2, 4), (6, 7), (1, 7)]
#     PROJECTIONS = [(4, 5), (2, 4), (6, 7), (1, 7)]
    f,p=plt.subplots(1,4,figsize=(20,3))
    for k, projection in enumerate(PROJECTIONS):

        if different_logsig is not None:
            p[k].scatter(different_logsig[:, projection[0]], different_logsig[:, projection[1]],\
                       label=labels[0], alpha = 0.7, color = 'tab:red')
        if different_logsig is None or plot_all is True:
            p[k].scatter(train_logsig[:, projection[0]], train_logsig[:, projection[1]],\
               label=labels[1], alpha = 0.7, color = 'tab:blue')
            
        
        p[k].scatter(test_logsig[:, projection[0]], test_logsig[:, projection[1]],\
                       label=labels[2], alpha = 0.7, color = 'tab:green')
        p[k].scatter(generated_logsig1[:, projection[0]], generated_logsig1[:, projection[1]],\
                     label=labels[3], alpha = 0.7, color = 'tab:orange')
        p[k].set_xlabel(logkeys[projection[0]], fontsize=10)
        p[k].set_ylabel(logkeys[projection[1]], fontsize=10)
        p[k].legend()
        p[k].grid()
    plt.suptitle('Log-signature')
    plt.show()
    #     plt.xticks([])
    #     plt.yticks([])
    
    

def _load_rough_bergomi(params, freq, ll = False):
    grid_points_dict = {"M": 28, "W": 10, "Y": 252}
    grid_points = grid_points_dict[freq]
    N = grid_points-1
    params["T"] = grid_points / grid_points_dict["Y"]
    paths = rough_bergomi(grid_points, **params)
    if ll:
        windows = np.array([leadlag(path) for path in paths])
        time = 0
    else:   
        time = np.linspace(0,1,N+1)
        path_exp = paths[:,:,None]
        path_exp = np.array([np.concatenate([time[:,None],path0],axis = -1) for path0 in path_exp])
        windows = path_exp
    return windows, paths, time


def _logsig(path,order):
    return esig.tosig.stream2logsig(path, order)
def _sig(path,order):
    return esig.tosig.stream2sig(path, order)

def get_data(order, params, freq, ll, scale):
    Batchsize = params['M']
    train_windows, train_path, time = _load_rough_bergomi(params, freq, ll)
    
    train_logsig = signatory.logsignature(torch.tensor(train_windows), order).numpy()
    train_sig = signatory.signature(torch.tensor(train_windows), order).numpy()
    train_sig_exp = np.concatenate([np.ones([Batchsize,1]),train_sig],axis = 1)
#     train_sig_normalized = normalize.normalize_sig(2, order, train_sig_exp)
    
    test_windows, test_path, time = _load_rough_bergomi(params, freq, ll)
    test_logsig = signatory.logsignature(torch.tensor(test_windows), order).numpy()
    test_sig = signatory.signature(torch.tensor(test_windows), order).numpy()
    test_sig_exp = np.concatenate([np.ones([Batchsize,1]),test_sig],axis = 1)
    if scale:
        scaler_logsig = MinMaxScaler(feature_range=(0.00001, 0.99999))
        logsig_transformed = scaler_logsig.fit_transform(train_logsig)
        data = logsig_transformed[1:]   # 1 week forecasting 1 week 
        data_cond = logsig_transformed[:-1] 
        data_cond = np.zeros_like(data_cond)         ######################################### for VAE
        scaler = scaler_logsig
    else:
        logsig_transformed = None
        scaler = None
        data = train_logsig[1:]   # 1 week forecasting 1 week 
        data_cond = train_logsig[:-1] 
        data_cond = np.zeros_like(data_cond)     ######################################### for VAE
        
    return train_windows, train_path, train_logsig, train_sig, train_sig_exp, test_windows, test_path,\
test_logsig, test_sig, test_sig_exp, logsig_transformed, data, data_cond, scaler

def inverse_plot(path, y_recover):
    f,p=plt.subplots(2,4,figsize=(16,4)) 
    Batchsize = path.shape[0]
    for i in range(2):
        for j in range(4):
            idx = np.random.randint(0,Batchsize)
            p[i,j].plot(path[idx,:,-1], label = 'True')
            p[i,j].plot(y_recover[idx,:,-1] + 10, label = 'Recover')
            p[i,j].grid()
            p[i,j].legend()
    plt.suptitle('Title')
    plt.show()
    
import torch
import neural_inverse
def ll_reverse(N,order,paths,logsigs):
    Recovered= []
    for p,logsig0 in zip(paths, logsigs):
        net0, y_recover0, logsig_recover0 = neural_inverse.inverse_leadlag_path(torch.tensor(logsig0[None,:]), 27, order, 1)
        Recovered.append(y_recover0[0,::2, 0].numpy()+10)
        print(len(Recovered))
#         plt.plot(p)
#         plt.plot(y_recover0[0,::2, 0].numpy()+10)
#         plt.legend(['True', 'Recover'])
#         plt.show()
    return Recovered


def ta_reverse(N,order,paths,logsigs):
    Recovered= []
    for p,ls in zip(paths, logsigs):
        net, y_recover, logsig_recover = neural_inverse.inverse_multiple_path_time(ls[None,:], N, order, 1)
        Recovered.append((y_recover[0,:,1]).numpy() + 10)
        print(len(Recovered))    
    return Recovered


import evaluation
import process_discriminator
from tqdm.auto import tqdm


def model_test_plot(model, order, params, params_diff, ll, scale, scaler):
    X_windows, X_path, X_logsig, X_sig, X_sig_exp, Y_windows, Y_path,\
    Y_logsig, Y_sig, Y_sig_exp, _, X_data, X_data_cond, _ = get_data(order, params, 'M', ll, scale)
    phi, PHI_X, PHI_Y = process_discriminator.T_global(X_sig_exp, Y_sig_exp, order=order, verbose=False, normalise=True, compute_sigs=False)
    TU0 = []
    TU1 = []
    TU2 = []
    for i in tqdm(range(20)):
        X_windows, X_path, X_logsig, X_sig, X_sig_exp, Y_windows, Y_path,\
        Y_logsig, Y_sig, Y_sig_exp, _, X_data, X_data_cond, X_scaler = get_data(order, params, 'M', ll, scale)
        Z_windows, Z_path, Z_logsig, Z_sig, Z_sig_exp, _, _,\
        _, _, _, _, _, _, _ = get_data(order, params_diff, 'M', ll, scale)

        _, X_generated_logsig = model.generate(torch.tensor(X_data_cond, dtype = torch.float))
        if scaler is not None:
            X_generated_logsig = scaler.inverse_transform(X_generated_logsig)
        X_generated_sig = np.array([tosig.logsig2sig(logsig, 2, order) for logsig in tqdm(X_generated_logsig)]) 

        plot_sig(order, X_sig_exp, Y_sig_exp, X_generated_sig, Z_sig_exp)
        plot_logsig(order, X_logsig, Y_logsig, X_generated_logsig, Z_logsig)

        result, TU = process_discriminator.test_fix(X_sig_exp[1:], Y_sig_exp[1:],\
                                                    order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
        TU0.append(TU)

        result, TU = process_discriminator.test_fix(X_sig_exp[1:], X_generated_sig,\
                                                    order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
        TU1.append(TU)

        result, TU = process_discriminator.test_fix(X_sig_exp[1:], Z_sig_exp[1:],\
                                                    order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
        TU2.append(TU)
    return TU0, TU1, TU2

def signature_MMD_test_plot(m,alpha,TU, label = None):
    K = 8
    split = 4*K / np.sqrt(m) * np.sqrt(np.log(1/alpha))
    if label is None:
        label = ['Same', 'Normal generator', 'Different'] 
    plt.figure(figsize = (16,3))
    a = plt.hist(TU.T, bins = 40, histtype='bar', stacked=True, label = label)
    plt.vlines(split,0,a[0].max(), color = 'r', label = 'acceptance: ' + str(alpha))
    plt.legend()
    plt.grid()
    plt.title('MMD test')
    # plt.ylabel('$T_{U}^{2}"$')
    plt.show()
    
    
def recoverpath_plot(test_path, Recovered_test0, Recovered_generate0):
    recover_path1 = np.array(Recovered_test0)
    recover_path2 = np.array(Recovered_generate0)
    f, p =plt.subplots(1,3,figsize = [15,3], sharey = True)
    p[0].plot(test_path.T, "k", alpha=0.1)
    p[1].plot(recover_path1.T, "r", alpha=0.1)
    p[2].plot(recover_path2.T, "b", alpha=0.1)
    p[0].title.set_text('Test paths')
    p[1].title.set_text('Path recovered from test log-signatures')
    p[2].title.set_text('Path recovered from generated log-signatures')
    for i in range(3):
        p[i].grid()
    plt.show()
    
import gpflow
from gpsig import kernels   
import tensorflow as tf

def lift_test(train_path, Recovered_test0, Recovered_generate0):
    num_levels = 3
    num_lags = 0
    kernE = kernels.SignatureKernel(base_kernel = gpflow.kernels.Exponential(), \
                                    num_levels = num_levels, order = num_levels, num_lags = num_lags)

    recover_path1 = np.array(Recovered_test0)
    recover_path2 = np.array(Recovered_generate0)
    X = tf.constant([leadlag(p) for p in train_path],'float64')
    Y = tf.constant([leadlag(p) for p in recover_path1],'float64')
    Z = tf.constant([leadlag(p) for p in recover_path2],'float64')
    H1_E = evaluation.Sig_TU2(X,Y,kernE)
    H2_E = evaluation.Sig_TU2(X,Z,kernE)
    return H1_E, H2_E

def lift_test_benchmark(order, params, params_diff, ll, scale):
    H0_E = []
    H3_E = []
    num_levels = 3
    num_lags = 0
    kernE = kernels.SignatureKernel(base_kernel = gpflow.kernels.Exponential(), \
                                    num_levels = num_levels, order = num_levels, num_lags = num_lags)
    for i in tqdm(range(20)):
        X_windows, X_path, X_logsig, X_sig, X_sig_exp, Y_windows, Y_path,\
        Y_logsig, Y_sig, Y_sig_exp, _, _, _, _ = get_data(order, params, 'M', ll, scale)
        Z_windows, Z_path, Z_logsig, Z_sig, Z_sig_exp, _, _,\
        _, _, _, _, _, _, _ = get_data(order, params_diff, 'M', ll, scale)
        X = tf.constant(X_windows[1:])
        Y = tf.constant(Y_windows[1:])
        Z = tf.constant(Z_windows[1:])
        TU = evaluation.Sig_TU2(X,Y,kernE)
        H0_E.append(TU)
        TU = evaluation.Sig_TU2(X,Z,kernE)
        H3_E.append(TU)
    return H0_E, H3_E

    
    
def time_aug(path):
    N = path.shape[0]-1
    time = np.linspace(0,1,N+1)
    windows = np.concatenate([time[:,None],path[:,None]],axis = -1) 
    return windows
    
    
    
def path_MMD_test_plot(H0,H1,H2):
    plt.figure(figsize = (16,3))
    label = ['Same', 'Different'] 
    H0 = np.array(H0)
    a = plt.hist(H0.T, bins = 40, density=True, histtype='bar', stacked=True, label = label, alpha = 0.8)
    plt.vlines(H1,0,a[0].max(), color = 'k', label = 'train paths VS paths from train log-signature')
    plt.vlines(H2,0,a[0].max(), color = 'r', label = 'train paths VS paths from generated log-signature')
    plt.title('MMD test')
    plt.legend()
    plt.grid()
    plt.show()


def plot_helper_2dpath(p,path_two,color):
    N = path_two.shape[0]-1
    for i in range(N):
        x0 = path_two[i,0]
        y0 = path_two[i,1]
        x1 = path_two[i+1,0] - path_two[i,0]
        y1 = path_two[i+1,1] - path_two[i,1]
        l = p.arrow(x0, y0, x1, y1,length_includes_head=True, head_width=.02, color = color)
    return l

def compare_ll_ta_plot(train_path, train_windows):
    f,p=plt.subplots(1,2,figsize=(16,3))
    batchsize = train_path.shape[0]
    idx = np.random.randint(0,batchsize)
    l1 = plot_helper_2dpath(p[0], leadlag(train_path[idx]),'tab:blue')
    l2 = plot_helper_2dpath(p[1], (train_windows[idx]),'tab:orange')
    p[0].legend((l1,), ['lead-lag transformation'])
    p[1].legend((l2,), ['Time augmentation'])
    for i in range(2):
        p[i].grid()
        p[i].set_xlabel('$X^{1}$')
        p[i].set_ylabel('$X^{2}$')
    plt.show()


    
    

def model_test_plot_tf(model1, model2, order, params, params_diff, ll, scale, scaler):
    X_windows, X_path, X_logsig, X_sig, X_sig_exp, Y_windows, Y_path,\
    Y_logsig, Y_sig, Y_sig_exp, _, X_data, X_data_cond, _ = get_data(order, params, 'M', ll, scale)
    phi, PHI_X, PHI_Y = process_discriminator.T_global(X_sig_exp, Y_sig_exp, order=order, verbose=False, normalise=True, compute_sigs=False)
    TU0 = []
    TU1_1 = []
    TU1_2 = []
    TU2 = []
    for i in tqdm(range(20)):
        X_windows, X_path, X_logsig, X_sig, X_sig_exp, Y_windows, Y_path,\
        Y_logsig, Y_sig, Y_sig_exp, _, X_data, X_data_cond, X_scaler = get_data(order, params, 'M', ll, scale)
        Z_windows, Z_path, Z_logsig, Z_sig, Z_sig_exp, _, _,\
        _, _, _, _, _, _, _ = get_data(order, params_diff, 'M', ll, scale)
        
        X_generated_logsig1 = model1.generate(X_data_cond)
        X_generated_logsig1 = scaler.inverse_transform(X_generated_logsig1)
        X_generated_sig1 = np.array([tosig.logsig2sig(logsig, 2, order) for logsig in tqdm(X_generated_logsig1)])
        
        X_generated_logsig2 = model2.generate(X_data_cond)
        X_generated_logsig2 = scaler.inverse_transform(X_generated_logsig2)
        X_generated_sig2 = np.array([tosig.logsig2sig(logsig, 2, order) for logsig in tqdm(X_generated_logsig2)])

        plot_sig(order, X_sig_exp, Y_sig_exp, X_generated_sig1, X_generated_sig2, True,\
                ["VAE Student", "Train data", "Test data", "VAE normal", ])
        plot_logsig(order, X_logsig, Y_logsig, X_generated_logsig1, X_generated_logsig2, True,\
                   ["VAE Student", "Train data", "Test data", "VAE normal", ])

        result, TU = process_discriminator.test_fix(X_sig_exp[1:], Y_sig_exp[1:],\
                                                    order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
        TU0.append(TU)

        result, TU = process_discriminator.test_fix(X_sig_exp[1:], X_generated_sig1,\
                                                    order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
        TU1_1.append(TU)
        
        result, TU = process_discriminator.test_fix(X_sig_exp[1:], X_generated_sig2,\
                                                    order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
        TU1_2.append(TU)

        result, TU = process_discriminator.test_fix(X_sig_exp[1:], Z_sig_exp[1:],\
                                                    order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
        TU2.append(TU)
    return TU0, TU1_1, TU1_2, TU2

    
    
    
    
    
    
    
    
    
    
    
    
    
    
# N = 27
# order = 4
# Recovered_train1 = []
# for p,ls in zip(train_path[1:], torch.tensor(train_logsig)[1:]):
#     net, y_recover, logsig_recover = neural_inverse.inverse_multiple_path_time(ls[None,:], N, order, 1)
#     Recovered_train1.append((y_recover[0,:,1]).numpy() + 10)
#     print(len(Recovered_train1))    
    
    
# reload(neural_inverse)
# N = 27
# order = 4
# Recovered_test1 = []
# for p,ls in zip(X_path[1:], torch.tensor(X_logsig)[1:]):
#     net, y_recover, logsig_recover = neural_inverse.inverse_multiple_path_time(ls[None,:], N, order, 1)
#     Recovered_test1.append((y_recover[0,:,1]).numpy() + 10)
#     print(len(Recovered_test1))
#     plt.plot(p[:])
#     plt.plot(y_recover[0,:,1] + 10)
#     plt.legend(['True', 'Recover'])
#     plt.show()    
    
    
# def hist_plot(order, params, params_diff):
    
#     X_windows, X_path, X_logsig, X_sig, X_sig_exp, Y_windows, Y_path,\
#     Y_logsig, Y_sig, Y_sig_exp, _, _, _, _ = get_data(order, params, 'M', False, False)
#     phi, PHI_X, PHI_Y = process_discriminator.T_global(X_sig, Y_sig, order=order, verbose=False, normalise=True, compute_sigs=False)
#     params_diff = {
#                 "M": 100,
#                 "H": 0.14,
#                 "rho": -0.85,
#                 "xi0": 0.5,
#                 "nu": 1.5,
#                 "S0": 10.
#               }
#     TU0 = []
#     TU1 = []
#     TU2 = []
#     for i in tqdm(range(20)):
#         X_windows, X_path, X_logsig, X_sig, X_sig_exp, Y_windows, Y_path,\
#         Y_logsig, Y_sig, Y_sig_exp, _, X_data, X_data_cond, _ = vae_helper.get_data(order, params, 'M', False, False)
#         Z_windows, Z_path, Z_logsig, Z_sig, Z_sig_exp, _, _,\
#         _, _, _, _, _, _, _ = vae_helper.get_data(order, params_diff, 'M', False, False)

#         _, X_generated_logsig = model.generate(torch.tensor(X_data_cond, dtype = torch.float))
#         X_generated_logsig = scaler.inverse_transform(X_generated_logsig)
#         X_generated_sig = np.array([tosig.logsig2sig(logsig, 2, order) for logsig in tqdm(X_generated_logsig)])

#         result, TU = process_discriminator.test_fix(X_sig_exp[1:], Y_sig_exp[1:],\
#                                                     order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
#         TU0.append(TU)

#         result, TU = process_discriminator.test_fix(X_sig_exp[1:], X_generated_sig,\
#                                                     order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
#         TU1.append(TU)

#         result, TU = process_discriminator.test_fix(X_sig_exp[1:], Z_sig_exp[1:],\
#                                                     order=order, confidence_level=0.99, phi_x = phi, compute_sigs=False)
#         TU2.append(TU)

    
    
    
    

