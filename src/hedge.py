from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

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

def delta_comapre(K,sigma,time,path0,y_recover0,y_recover1,y_recover_time_0 ):
    C, V = delta_hedge(K, sigma, time, path0[0,:,0])
    C_recover0, V_recover0 = delta_hedge(K, sigma, time, y_recover0.numpy())
    C_recover_time_0, V_recover_time_0 = delta_hedge(K, sigma, time, y_recover_time_0.numpy())
    C_recover1, V_recover1 = delta_hedge(K, sigma, time, y_recover1)
    f,p=plt.subplots(1,2,figsize=(16,4)) 
    p[0].plot(path0[0])
    p[0].plot(y_recover0)
    p[0].plot(y_recover_time_0)
    p[0].plot(y_recover1)
    p[0].legend(['True', 'Neural(lead-lag)', 'Neural(time-aug)', 'Evolution(lead-lag)'])
    p[1].plot(V[:-1])
    p[1].plot(V_recover0[:-1])
    p[1].plot(V_recover_time_0[:-1])
    p[1].plot(V_recover1[:-1])
    p[1].plot(C)
    p[1].legend(['True', 'Neural(lead-lag)', 'Neural(time-aug)', 'Evolution(lead-lag)', 'Option price'])
    p[0].title.set_text('Stock price: $S_{t}$')
    p[1].title.set_text('Replication portfolio: $V_{t}$')
    
    for i in range(2):
        p[i].grid()
    plt.show()
    return V_recover0, V_recover1