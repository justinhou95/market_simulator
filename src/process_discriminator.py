import numpy as np
from esig import tosig
from tqdm.auto import tqdm
from scipy.optimize import brentq
from joblib import Parallel, delayed
import ast

# max_norm_square = 2.


def Sig_TU2(X,Y,kern):
    kxx = kern(X,X)
    kxy = kern(X,Y)
    kyy = kern(Y,Y)
    m = X.shape[0]
    n = Y.shape[0]
    TU = 0.
    TU += (np.sum(kxx) - np.trace(kxx)) / (m * (m-1))
    TU += (np.sum(kyy)-np.trace(kyy)) / (n * (n-1))
    TU -= 2 * np.sum(kxy) / (m * n)
    return TU

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

def phi(x, order, keys):
    x = np.array(x)

    a = x ** 2
    a[0] -= psi(norm(x))

#     a[0] -= max_norm_square
    
    
    f = lambda z: np.dot(a, [z ** (2 * len(keys[i])) for i in range(len(a))])

    return brentq(f, 0, 100)


def get_keys(dim, order):
    s = tosig.sigkeys(dim, order)

    tuples = []

    for t in s.split():
        if len(t) > 2:
            t = t.replace(")", ",)")

        tuples.append(ast.literal_eval(t))

    return tuples

def Phi(X, order, normalise=True, compute_sigs=True):
    if compute_sigs:
        dim = np.shape(X)[1]
        sig = tosig.stream2sig(np.array(X), order)
    else:
        dim = 2
        sig = np.array(X)
    print('before normalized')
    print(np.linalg.norm(sig)**2)
    if not normalise:
        return sig    
#     print(np.linalg.norm(sig)**2)
    keys = get_keys(dim, order)

    phi_x = phi(tuple(sig), order, keys)
    Lambda = np.array([phi_x ** len(t) for t in keys])
    print('after normalized')
    print(np.linalg.norm(Lambda * sig)**2)
    
    return Lambda * sig

def T(set1, set2, order, verbose=True, normalise=True, compute_sigs=True):
    m = len(set1)
    n = len(set2)

    X = Parallel(n_jobs=1)(delayed(Phi)(path, order, normalise, compute_sigs) for path in tqdm(set1, desc="Computing signatures of population 1", disable=(not verbose)))
    Y = Parallel(n_jobs=1)(delayed(Phi)(path, order, normalise, compute_sigs) for path in tqdm(set2, desc="Computing signatures of population 2", disable=(not verbose)))

    XX = np.dot(X, np.transpose(X))
    YY = np.dot(Y, np.transpose(Y))
    XY = np.dot(X, np.transpose(Y))

    TU = 0.
    TU += XX.sum() / (m * m)
    TU += YY.sum() / (n * n)
    TU -= 2 * XY.sum() / (m * n)


    return TU


def Phi_global(X, order, normalise=True, compute_sigs=True):
    if compute_sigs:
        dim = np.shape(X)[1]
        sig = tosig.stream2sig(np.array(X), order)
    else:
        dim = 2
        sig = np.array(X)
    keys = get_keys(dim, order)
    phi_x = phi(tuple(sig), order, keys)
    
    return sig, phi_x

    

def T_global(set1, set2, order, verbose=True, normalise=True, compute_sigs=True):
    
    
    
    m = len(set1)
    n = len(set2)

    X_pre = Parallel(n_jobs=1)(delayed(Phi_global)(path, order, normalise, compute_sigs) for path in tqdm(set1, desc="Computing signatures of population 1", disable=(not verbose)))
    Y_pre = Parallel(n_jobs=1)(delayed(Phi_global)(path, order, normalise, compute_sigs) for path in tqdm(set2, desc="Computing signatures of population 2", disable=(not verbose)))
    
    PHI_X = []
    X = []
    for x in X_pre:
        X.append(x[0])
        PHI_X.append(x[1])
    min_phi_x = min(PHI_X)    
    
    PHI_Y = []
    Y = []
    for y in Y_pre:
        Y.append(y[0])
        PHI_Y.append(y[1])
    min_phi_y = min(PHI_Y)
    
    min_phi = min(min_phi_x, min_phi_y)
    
    print('='*100)
    print(min_phi)
    print('='*100)
    
    return min_phi, PHI_X, PHI_Y


def Phi_fix(X, order, normalise=True, compute_sigs=True, phi_x = 1):
#     print(phi_x)
#     print('not fix')
    if compute_sigs:
        dim = np.shape(X)[1]
        sig = tosig.stream2sig(np.array(X), order)
    else:
        dim = 2
        sig = np.array(X)
    if not normalise:
        return sig
    
    keys = get_keys(dim, order)
    Lambda = np.array([phi_x ** len(t) for t in keys])
    
    sig_now = Lambda * sig
    
    phi_x_now = phi(tuple(sig_now), order, keys)
    Lambda_now = np.array([phi_x_now ** len(t) for t in keys])
    
    return Lambda_now * sig_now






def T_fix(set1, set2, order, verbose=True, normalise=True, compute_sigs=True, phi_x = 1):
    m = len(set1)
    n = len(set2)
    
    X = Parallel(n_jobs=1)(delayed(Phi_fix)(path, order, normalise, compute_sigs, phi_x) for path in tqdm(set1, desc="Computing signatures of population 1", disable=(not verbose)))
    Y = Parallel(n_jobs=1)(delayed(Phi_fix)(path, order, normalise, compute_sigs, phi_x) for path in tqdm(set2, desc="Computing signatures of population 2", disable=(not verbose)))

    XX = np.dot(X, np.transpose(X))
    YY = np.dot(Y, np.transpose(Y))
    XY = np.dot(X, np.transpose(Y))

    TU = 0.
    TU += XX.sum() / (m * m)
    TU += YY.sum() / (n * n)
    TU -= 2 * XY.sum() / (m * n)
    


    return TU


def c_alpha(m, alpha):
    K = 8.
    return 4 * np.sqrt(-np.log(alpha) / m)
    return (2 * K / m) * (1 + np.sqrt(-2 * np.log(alpha))) ** 2

def test(set1, set2, order, confidence_level=0.99, **kwargs):

    assert len(set1) == len(set2), "Same size samples accepted for now."

    assert confidence_level >= 0. and confidence_level <= 1., "Confidence level must be in [0, 1]."

    m = len(set1)

    TU = T(set1, set2, order, **kwargs)
    c = c_alpha(m, confidence_level)
    print('c:', c)
    print('TU:', TU)
    print('m:',m)

    return TU > c, TU


def test_global(set1, set2, order, confidence_level=0.99, **kwargs):

    assert len(set1) == len(set2), "Same size samples accepted for now."

    assert confidence_level >= 0. and confidence_level <= 1., "Confidence level must be in [0, 1]."

    m = len(set1)

    TU = T_global(set1, set2, order, **kwargs)
    c = c_alpha(m, confidence_level)
    print('c:', c)
    print('TU:', TU)
    print('m:',m)

    return TU > c, TU



def test_fix(set1, set2, order, confidence_level=0.99, **kwargs):

    assert len(set1) == len(set2), "Same size samples accepted for now."

    assert confidence_level >= 0. and confidence_level <= 1., "Confidence level must be in [0, 1]."

    m = len(set1)

    TU = T_fix(set1, set2, order, **kwargs)
    c = c_alpha(m, confidence_level)
    print('c:', c)
    print('TU:', TU)
    print('m:',m)

    return TU > c, TU
