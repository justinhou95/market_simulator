import numpy as np
import base
from esig import tosig
from tqdm.auto import tqdm
from scipy.optimize import brentq
from joblib import Parallel, delayed
import ast
import torch
import signatory
from rough_bergomi import rough_bergomi


def transform(sig, phi_x, keys):
    Lambda = np.array([phi_x ** len(t) for t in keys])
    return Lambda * sig

def psi(x, M=4, a=1):
    x = x ** 2
    if x <= M:
        return x
    return M + M ** (1 + a) * (M ** (-a) - x ** (-a)) / a

def norm(x):
    return np.linalg.norm(x)

def phi(x, order, power):
    a = x ** 2
    a[0] -= psi(norm(x))    
    f = lambda z: np.dot(a, z ** (2 * power))
    return brentq(f, 0, 100)

def Phi(sig, dim, order, phi_x = None):
    power = np.array([0] + [len(w) for w in signatory.all_words(dim,order)])
    if phi_x is None:
        phi_x = phi(sig, order, power)
    Lambda = phi_x ** power
    output = Lambda * sig
    return output

def normalize_sig(dim,order,train_sig_exp):
    train_sig_normalized = np.array([Phi(s, dim, order) for s in train_sig_exp])
    return train_sig_normalized

def normalize_logsig0(dim,order, logsig):
    a = signatory.lyndon_words(dim,order)
    power = np.array([len(w) for w in a])
    phi_x = 10
    Lambda = phi_x ** power
    output = Lambda * logsig
    return output

def normalize_logsig(dim, order, logsig):
    logsig_normalized = np.array([normalize_logsig0(dim, order,s) for s in logsig])
    return logsig_normalized
    
    
