from scipy.stats import norm
import numpy as np

def BlackScholes(tau, S, K, sigma, index = None):
    d1=np.log(S/K)/sigma/np.sqrt(tau)+0.5*sigma*np.sqrt(tau)
    d2=d1-sigma*np.sqrt(tau)
    npv=(S*norm.cdf(d1)-K*norm.cdf(d2))
    delta=norm.cdf(d1)
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(tau))
    vega=S*norm.pdf(d1)*np.sqrt(tau)
    theta=-.5*S*norm.pdf(d1)*sigma/np.sqrt(tau)
    data = {'npv':npv,'delta':delta,'gamma':gamma,'vega':vega,'theta':theta}
    if index:
        return data[index]
    else:
        return data
    
def delta_hedge(K, sigma, time, S):   # special for T = 1 and Bt = t
    N = len(time)-1
    dt = 1/N
    data = [BlackScholes(1-t, s, K, sigma) for t,s in zip(time[:-1],S[:-1])]
    C = np.array([d['npv'] for d in data])
    delta = np.array([d['delta'] for d in data])
    theta = np.array([d['theta'] for d in data])
    gamma = np.array([d['gamma'] for d in data])
    V = np.zeros(N+1)
    S_holding = delta
    B_holding = theta + 0.5 * gamma * S[:-1]**2
    V[0] = C[0]
    for i in range(N):
        V[i+1] = V[i] + B_holding[i]*dt + S_holding[i]*(S[i+1]-S[i])
    return C, V