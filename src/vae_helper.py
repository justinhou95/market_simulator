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




def plot_sig(order, train_sig, test_sig, generated_sig1,different_sig = None):

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
                       label="Different data", alpha = 0.7, color = 'tab:red')
        else:
            p[k].scatter(train_sig[:, projection[0]], train_sig[:, projection[1]],\
                       label="Train data", alpha = 0.7, color = 'tab:blue')
         
        p[k].scatter(test_sig[:, projection[0]], test_sig[:, projection[1]],\
                       label="Test data", alpha = 0.7, color = 'tab:green')
        p[k].scatter(generated_sig1[:, projection[0]], generated_sig1[:, projection[1]],\
                         label="VAE normal", alpha = 0.7, color = 'tab:orange')
        
        p[k].set_xlabel(keys[projection[0]], fontsize=10)
        p[k].set_ylabel(keys[projection[1]], fontsize=10)
        p[k].legend()
        p[k].grid()
    plt.suptitle('Signature')
    plt.show()
    
def plot_logsig(order, train_logsig, test_logsig, generated_logsig1, different_logsig = None):
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
                       label="Different data", alpha = 0.7, color = 'tab:red')
        else:
            p[k].scatter(train_logsig[:, projection[0]], train_logsig[:, projection[1]],\
               label="Train data", alpha = 0.7, color = 'tab:blue')
            
        
        p[k].scatter(test_logsig[:, projection[0]], test_logsig[:, projection[1]],\
                       label="Test data", alpha = 0.7, color = 'tab:green')
        p[k].scatter(generated_logsig1[:, projection[0]], generated_logsig1[:, projection[1]],\
                     label="VAE normal", alpha = 0.7, color = 'tab:orange')
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
    train_windows, train_path, time = _load_rough_bergomi(params, freq, ll)
    train_logsig = signatory.logsignature(torch.tensor(train_windows), order).numpy()
    train_sig = signatory.signature(torch.tensor(train_windows), order).numpy()
    train_sig_exp = np.concatenate([np.ones([100,1]),train_sig],axis = 1)
#     train_sig_normalized = normalize.normalize_sig(2, order, train_sig_exp)
    
    test_windows, test_path, time = _load_rough_bergomi(params, freq, ll)
    test_logsig = signatory.logsignature(torch.tensor(test_windows), order).numpy()
    test_sig = signatory.signature(torch.tensor(test_windows), order).numpy()
    test_sig_exp = np.concatenate([np.ones([100,1]),test_sig],axis = 1)
    if scale:
        scaler_logsig = MinMaxScaler(feature_range=(0.00001, 0.99999))
        logsig_transformed = scaler_logsig.fit_transform(train_logsig)
        data = logsig_transformed[1:]   # 1 week forecasting 1 week 
        data_cond = logsig_transformed[:-1] 
        data_cond = np.zeros_like(data_cond)
        scaler = scaler_logsig
    else:
        logsig_transformed = None
        scaler = None
        data = train_logsig[1:]   # 1 week forecasting 1 week 
        data_cond = train_logsig[:-1] 
        data_cond = np.zeros_like(data_cond)
        
    return train_windows, train_path, train_logsig, train_sig, train_sig_exp, test_windows, test_path,\
test_logsig, test_sig, test_sig_exp, logsig_transformed, data, data_cond, scaler

def inverse_plot(path, y_recover):
    f,p=plt.subplots(2,4,figsize=(16,4)) 
    for i in range(2):
        for j in range(4):
            idx = np.random.randint(0,100)
            p[i,j].plot(path[idx,:,-1], label = 'True')
            p[i,j].plot(y_recover[idx,:,-1] + 10, label = 'Recover')
            p[i,j].grid()
            p[i,j].legend()
    plt.suptitle('Title')
    plt.show()
    
    
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

    
    
    
    

