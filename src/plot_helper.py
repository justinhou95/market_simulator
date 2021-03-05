import matplotlib.pyplot as plt
from esig.tosig import sigkeys
import numpy as np
from esig.tosig import logsigkeys
import re


def plot_sig(order, train_sig, test_sig, generated_sig1):

    keys = sigkeys(2, order).split()
    factor = []
    for i in keys:
        factor.append((np.ceil((len(i)-2)/2)))
    factor = np.array(factor)

    PROJECTIONS = [(4, 5), (3, 4), (7, 8), (9, 10)]
    f,p=plt.subplots(1,4,figsize=(16,4))
    for k, projection in enumerate(PROJECTIONS):
        p[k].scatter(train_sig[:, projection[0]], train_sig[:, projection[1]],\
                       label="Train data", alpha = 0.7, color = 'tab:blue')
        p[k].scatter(test_sig[:, projection[0]], test_sig[:, projection[1]],\
                       label="Test data", alpha = 0.7, color = 'tab:green')
        p[k].scatter(generated_sig1[:, projection[0]], generated_sig1[:, projection[1]],\
                         label="CVAE normal (tfp)", alpha = 0.7, color = 'tab:orange')
        p[k].set_xlabel(keys[projection[0]], fontsize=10)
        p[k].set_ylabel(keys[projection[1]], fontsize=10)
        p[k].legend()
        p[k].grid()
    plt.show()
    
def plot_logsig(order, train_logsig, test_logsig, generated_logsig1):
    logkeys = logsigkeys(2, order).split()
    logfactor = []
    for i in logkeys:
        logfactor.append((len(re.sub("\D", "", i))))
    logfactor = np.array(logfactor)
    PROJECTIONS = [(4, 5), (2, 4), (6, 7), (1, 7)]
    f,p=plt.subplots(1,4,figsize=(16,4))
    for k, projection in enumerate(PROJECTIONS):
        p[k].scatter(train_logsig[:, projection[0]], train_logsig[:, projection[1]],\
                       label="Train data", alpha = 0.7, color = 'tab:blue')
        p[k].scatter(test_logsig[:, projection[0]], test_logsig[:, projection[1]],\
                       label="Test data", alpha = 0.7, color = 'tab:green')
        p[k].scatter(generated_logsig1[:, projection[0]], generated_logsig1[:, projection[1]],\

                     label="CVAE normal (tfp)", alpha = 0.7, color = 'tab:orange')
        p[k].set_xlabel(logkeys[projection[0]], fontsize=10)
        p[k].set_ylabel(logkeys[projection[1]], fontsize=10)
        p[k].legend()
        p[k].grid()
    plt.show()
    #     plt.xticks([])
    #     plt.yticks([])

