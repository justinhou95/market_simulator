import numpy as np
import Sig_method
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def nilpotent(M):
    B = np.zeros((M,M))
    for i in range(2,M):
        B[i,i-1]=1.0
    return B

def canonical(i,M):
    e = np.zeros((M,1))
    e[i,0]=1.0
    return e

def randomAbeta(d,M):
    A = []
    beta = []
    for i in range(d):
        B = 0.0*nilpotent(M) + np.random.normal(0.0,0.05,size=(M,M)) 
        B = np.random.permutation(B)
        A = A + [B]
        beta = beta + [0.0*canonical(i,M)+np.random.normal(0.0,0.05,size=(M,1))]
    return [A,beta]

def sigmoid(x):
    return np.tanh(x)

class resevior_dynamic:
    def __init__(self,d,M):
        self.M = M 
        self.d = d
        self.A, self.beta = randomAbeta(d,M)
        
    def reservoirfield(self,state,increment):
        value = np.zeros((self.M,1))
        for i in range(self.d):
            value = value + sigmoid(np.matmul(self.A[i],state) + self.beta[i])*increment[i]
        return value
        
def reservoir(BMpath, r, initialvalue = None):
    timesteps = BMpath.shape[0]
    d_R = r.M
    d = r.d
    helper = canonical(0,d_R)
    if initialvalue:
        helper[:d,0] = initialvalue
    reservoirpath = [helper]
    for i in range(timesteps-1):
        increment = BMpath[i+1]-BMpath[i]
        state = reservoirpath[-1]
        reservoirpath.append( state + r.reservoirfield(state,increment) + np.exp(-0.1*state)*(1/timesteps) )
    return np.squeeze(np.array(reservoirpath))    
    
def cut_path(path,sublength):
    start = 0
    end = start + sublength 
    path_split = []
    while start < path.shape[0]-5:
        path_split.append(path[start:end])
        start += sublength
        end = start + sublength
    return np.array(path_split)   
    
def Nonlinear(dimR, dim, alpha):
    # Compute weight
    inputs_initial = keras.Input(shape=(dim,))
    l_weigth_1 = tf.keras.layers.Dense(units = 10, activation='relu')
    l_weigth_2 = tf.keras.layers.Dense(units = dim*dim*dimR, activation='linear')
    l_reshape_1 = tf.keras.layers.Reshape((dimR, dim, dim), input_shape=(dim*dim*dimR,))
    weight = l_reshape_1(l_weigth_2(l_weigth_1(inputs_initial)))

    l_bias_1 = tf.keras.layers.Dense(units = 10, activation='relu')
    l_bias_2 = tf.keras.layers.Dense(units = dim, activation='linear')
    bias = l_bias_2(l_bias_1(inputs_initial))

    # Linear regression
    inputs_X = keras.Input(shape=(dimR,dim,1))
    x = tf.matmul(weight,inputs_X)
    outputs = tf.reduce_sum(x,axis = [1,3]) + bias

    model = tf.keras.Model(inputs = [inputs_initial, inputs_X], outputs = outputs)
    model.add_loss(alpha * tf.reduce_mean(tf.reduce_mean(tf.math.square(weight), axis = [1,2,3]), axis = 0))
    model.add_loss(alpha * tf.reduce_mean(tf.reduce_mean(tf.math.square(bias), axis = [1]), axis = 0))
    
    return model         
        
        
class Resevoir_split:
    def __init__(self, data, depth_learn, R):
        self.data = data 
        self.BMpath, self.SDEpath = data
        self.initial = self.SDEpath[0,:]
        self.d = self.SDEpath.shape[-1]
        self.d_BM = self.BMpath.shape[-1]
        self.timesteps = self.BMpath.shape[0]
        self.depth_learn = depth_learn
        self.R = R
        self.r = resevior_dynamic(self.d,R)
           
    def prepare(self, BMpath, SDEpath):
        initial = SDEpath[0,:]
        Y = SDEpath
        X0 = reservoir(BMpath, self.r)
        X0 = np.tensordot(X0,initial,axes = 0)
        X = np.reshape(X0,[-1,X0.shape[-1]*X0.shape[-2]])  
        X_sig_all = []
        for depth in self.depth_learn:
            X0 = Sig_method.sig_stream2(BMpath,depth)[0,:,:].numpy()
            X0 = np.tensordot(X0,initial,axes = 0)
            X_sig = np.reshape(X0,[-1,X0.shape[-1]*X0.shape[-2]])  
            X_sig_all.append(X_sig)
        return [Y,X] + X_sig_all
    
    def prepare_all(self):
        data = self.prepare(self.BMpath, self.SDEpath)
        self.Y, self.X, self.X_sig_all = data[0], data[1], data[2:]

    def prepare_split(self, sublength, trainto):
        self.sublength = sublength
        # (batch, sublength, dimensionBM)   i.e. (2,500,2)  non-intersection between different subpath
        self.BMpath_split = cut_path(self.BMpath[:trainto+1],sublength)  
        self.SDEpath_split = cut_path(self.SDEpath[:trainto+1],sublength)    # (batch, sublength, dimension)
        self.initial_split = np.array([path[0,:] for path in self.SDEpath_split]) # (batch, dimension)
        data_split = [self.prepare(BMpath, SDEpath) for BMpath, SDEpath in zip(self.BMpath_split, self.SDEpath_split)]          
        self.Ytrain0 = np.array([data[0] for data in data_split])
        self.Ytrain = np.reshape(self.Ytrain0,[-1,self.d])
        print(self.Ytrain0.shape,end=', ')
        self.Xtrain0 = np.array([data[1] for data in data_split])
        self.Xtrain = np.reshape(self.Xtrain0,[-1,self.Xtrain0.shape[-1]])
        print(self.Xtrain0.shape,end=', ')
        self.X_sig_train_all = []
        for i,depth in enumerate(self.depth_learn):
            X_train_sig0 = np.array([data[i+2] for data in data_split])
            X_train_sig = np.reshape(X_train_sig0,[-1,X_train_sig0.shape[-1]])
            self.X_sig_train_all.append(X_train_sig)
            print(np.shape(X_train_sig0),end=', ')
        print('') 
        
    def train(self, alpha, verbose = False, fix_intercept = False):
        self.lm = linear_model.Ridge(alpha = alpha, fit_intercept = fix_intercept)   
        self.lm.fit(self.Xtrain,self.Ytrain)
        if verbose:
            print('score: ',self.model.score(self.Xtrain,self.Ytrain))
            print('max coefficient: ',np.max(np.abs(self.model.coef_)))
        self.lm_all = []
        for Xtrain_sig in self.X_sig_train_all:
            lm_sig = linear_model.Ridge(alpha = alpha, fit_intercept = fix_intercept)
            lm_sig.fit(Xtrain_sig,self.Ytrain)
            self.lm_all.append(lm_sig)
            if verbose:
                print('score: ',model_sig.score(Xtrain_sig,self.Ytrain))
                print('max coefficient: ',np.max(np.abs(model_sig.coef_)))
                print('alpha： ',lm_sig.alpha_)

    def predict(self, X, X_sig_all):
        Y_predict = self.lm.predict(X)
        Y_predict_sig_all = [lm_sig.predict(X_sig) for lm_sig, X_sig in zip(self.lm_all, X_sig_all)]
        return Y_predict, Y_predict_sig_all
    
    def evaluate(self, Y_predict, Y_predict_sig_all, Y):
        DIFF = np.zeros([len(self.depth_learn)+1,self.d,Y_predict.shape[0]])
        ERROR = np.zeros([len(self.depth_learn)+1,self.d])
        for i in range(self.d):
            for j,Y_pre in enumerate(Y_predict_sig_all + [Y_predict]):
                diff = np.abs(Y_pre[:,i] - Y[:,i])
                error = np.max(diff)
                DIFF[j,i] = diff
                ERROR[j,i] = error
        return DIFF, ERROR
        
    def plot(self, DIFF, ERROR, Y_predict, Y_predict_sig_all, Y, name, verbose):
        f,p=plt.subplots(1,2,figsize=(16,3)) 
        for i in range(self.d):
            for j,Y_pre in enumerate(Y_predict_sig_all + [Y_predict]):
                p[i].plot(Y_pre[:,i]) 
            p[i].plot(Y[:,i])
            p[i].grid()
            p[i].legend(self.depth_learn +  ['Res','True'], loc = 'upper left')
            p[0].title.set_text('Stimulated path: $X^{1}_{t}$')
            p[1].title.set_text('Stimulated path: $X^{2}_{t}$')
            if name == 'all':
                p[i].vlines(1000,min(Y_pre[:,i].min(),Y[:,i].min())-0.1,min(Y_pre[:,i].max(),Y[:,i].max())+0.1,'r')
                plt.suptitle('Training path')
            else:
                plt.suptitle('Testing path')
        plt.show()
        if verbose == True:
            f,p=plt.subplots(1,2,figsize=(16,3))
            for i in range(2):
                [p[i].plot(diff) for diff in DIFF[:,i,:]]
                p[i].legend(self.depth_learn + ['Res'])
            plt.show()
        if verbose == True or name == 'test':    
            f,p=plt.subplots(1,2,figsize=(16,3))
            for i in range(2):
                p[i].plot(self.depth_learn, ERROR[:-1,i],'o-')
                p[i].set_yscale('log')
                p[i].set_xlabel('Order')
                p[i].set_ylabel('Uniform Error')
                p[i].grid()
            p[0].title.set_text('Uniform error: $X^{1}_{t}$')
            p[1].title.set_text('Uniform error: $X^{2}_{t}$')
            plt.suptitle('Uniform error of testing path')
            plt.show()
            
    def plot_all(self,plot = False, verbose = False):     
        self.Y_predict, self.Y_predict_sig_all = self.predict(self.X,self.X_sig_all)
        self.DIFF, self.ERROR = self.evaluate(self.Y_predict, self.Y_predict_sig_all, self.Y) 
        if plot:
            self.plot(self.DIFF, self.ERROR, self.Y_predict, self.Y_predict_sig_all, self.Y, 'all', verbose)
            
    def plot_test(self,plot = False, verbose = False):     
        self.Ytest_predict, self.Ytest_predict_sig_all = self.predict(self.Xtest,self.Xtest_sig_all)
        self.DIFFtest, self.ERRORtest = self.evaluate(self.Ytest_predict, self.Ytest_predict_sig_all, self.Ytest) 
        if plot:
            self.plot(self.DIFFtest, self.ERRORtest, \
                      self.Ytest_predict, self.Ytest_predict_sig_all, self.Ytest, 'test', verbose)
        
    
    def prepare_test(self, data_test, validto):
        self.data_test = data_test
        self.BMpath_test, self.SDEpath_test = self.data_test[0][:validto], self.data_test[1][:validto]
        data = self.prepare(self.BMpath_test, self.SDEpath_test)
        self.Ytest, self.Xtest, self.Xtest_sig_all = data[0], data[1], data[2:]
        
    def prepare_test_set(self, validto, data_test_set):
        self.data_test_set = data_test_set
        set_size = len(data_test_set)
        self.Ytest_set = []
        self.Xtest_set = []
        self.Xtest_sig_all_set = []
        for data_test in tqdm(data_test_set):
            BMpath_test, SDEpath_test = data_test[0][:validto], data_test[1][:validto]
            data = self.prepare(BMpath_test, SDEpath_test)
            Ytest, Xtest, Xtest_sig_all = data[0], data[1], data[2:]    
            self.Ytest_set.append(Ytest)
            self.Xtest_set.append(Xtest)
            self.Xtest_sig_all_set.append(Xtest_sig_all)       

    def test_in_set(self):
        res_predict_set = []
        sig_predict_set = []
        for Xtest, X_sig_all in zip(self.Xtest_set, self.Xtest_sig_all_set):
            Y_predict, Y_predict_sig_all = self.predict(Xtest, X_sig_all)
            res_predict_set.append(Y_predict)
            sig_predict_set.append(Y_predict_sig_all)
        res_predict_set = tf.convert_to_tensor(res_predict_set)
        sig_predict_set = tf.convert_to_tensor(sig_predict_set)
        self.X_res = res_predict_set
        self.X_sig =  [sig_predict_set[:,i,:,:] for i in range(len(self.depth_learn))]
            
 
        
        
        
        
        
        


        
        
        
        
# Non linear Non linear Non linear Non linear Non linear Non linear Non linear Non linear Non linear Non linear  




#     def train_split_nonlinear(self, alpha):
        
#         dimR = self.Xtrain.shape
#         self.lm = Nonlinear(dimR, dim, alpha = alpha) 
#         self.model = self.lm
#         self.model = self.lm.fit([self.Xtrain_nonlinear, Xtrain1],self.Ytrain)
#         if verbose:
#             print('score: ',self.model.score(self.Xtrain,self.Ytrain))
#             print('max coefficient: ',np.max(np.abs(self.model.coef_)))
#         self.lm_all = []
#         self.model_all = []
#         for Xtrain_sig in self.X_sig_train_all:
#             lm_sig = linear_model.Ridge(alpha = alpha, fit_intercept = fix_intercept)
#             model_sig = lm_sig.fit(Xtrain_sig,self.Ytrain)
#             self.lm_all.append(lm_sig)
#             self.model_all.append(model_sig)
#             if verbose:
#                 print('score: ',model_sig.score(Xtrain_sig,self.Ytrain))
#                 print('max coefficient: ',np.max(np.abs(model_sig.coef_)))
#                 print('alpha： ',lm_sig.alpha_)


        
        
    
        
        
        
        
        
     
        

# class Resevoir:
#     def __init__(self, sde, training):
#         self.sde = sde
#         self.training = training
#         self.BMpath=np.array(self.training[0])
#         self.SDEpath=np.array(self.training[1])
        
#     def prepare(self, depth_learn, traindtype = 'all'):
#         self.traindtype = traindtype
#         self.depth_learn = depth_learn
        
#         self.BMpath=self.training[0]
#         Y = self.training[1]
#         self.Ydata = np.squeeze(Y)
#         self.Ydatadiff = np.diff(self.Ydata,axis=0)
#         if traindtype == 'all':
#             self.Ytrain = np.concatenate((self.Ydata[:1000],self.Ydatadiff[:1000:1]),axis=0)
#         elif traindtype == 'data':
#             self.Ytrain = self.Ydata[:1000]
#         elif traindtype == 'diff':
#             self.Ytrain = self.Ydatadiff[:1000:1]
#         print(np.shape(self.Ytrain))
        
        
#         X=self.sde.reservoir(np.array(self.BMpath))
#         np.shape(X)
#         self.Xdata = np.squeeze(X)
#         self.Xdatadiff = np.diff(self.Xdata,axis=0)
#         if traindtype == 'all':
#             self.Xtrain = np.concatenate((self.Xdata[:1000],self.Xdatadiff[:1000:1]),axis=0)
#         elif traindtype == 'data':
#             self.Xtrain = self.Xdata[:1000]
#         elif traindtype == 'diff':
#             self.Xtrain = self.Xdatadiff[:1000:1]
#         print(np.shape(self.Xtrain))

#         self.X_sig_all = []
#         self.X_sig_train_all = []
#         self.X_sig_all_diff = []
#         for depth in self.depth_learn:
#             sig_path_stream = Sig_method.sig_stream2(np.array(self.BMpath),depth)[0,:,:].numpy()
#             Xdata_sig = sig_path_stream
#             Xdatadiff_sig = np.diff(Xdata_sig,axis=0)
#             if traindtype == 'all':
#                 Xtrain_sig = np.concatenate((Xdata_sig[:1000],Xdatadiff_sig[:1000:1]),axis=0)
#             elif traindtype == 'data':
#                 Xtrain_sig = Xdata_sig[:1000]
#             elif traindtype == 'diff':
#                 Xtrain_sig = Xdatadiff_sig[:1000:1]
#             print(np.shape(Xtrain_sig),end = '')
#             self.X_sig_all.append(Xdata_sig)
#             self.X_sig_train_all.append(Xtrain_sig)
#             self.X_sig_all_diff.append(Xdatadiff_sig)
#         print('')
            
#     def train(self):
#         alphas=np.logspace(-10, -9, 2)
#         cv = 10
#         self.lm = linear_model.RidgeCV(alphas = alphas, cv = cv)
#         self.model = self.lm.fit(self.Xtrain,self.Ytrain)
#         print('score: ',self.model.score(self.Xtrain,self.Ytrain))
#         print('max coefficient: ',np.max(np.abs(self.model.coef_)))
#         self.lm_all = []
#         self.model_all = []
#         for Xtrain_sig in self.X_sig_train_all:
#             lm_sig = linear_model.RidgeCV(alphas = alphas, cv = cv)
#             model_sig = lm_sig.fit(Xtrain_sig,self.Ytrain)
#             print('score: ',model_sig.score(Xtrain_sig,self.Ytrain))
#             print('max coefficient: ',np.max(np.abs(model_sig.coef_)))
#             self.lm_all.append(lm_sig)
#             self.model_all.append(model_sig)
#             print('alpha： ',lm_sig.alpha_)
            
            
         
#     def plot_train(self,verbose = False):
#         self.initial = self.Ydata[0,:]
#         f,p=plt.subplots(1,2,figsize=(16,3))
#         self.DIFF = []
#         self.ERROR = []
#         for i in range(2):
#             for model_sig, Xdata_sig, Xdatadiff_sig in zip(self.model_all, self.X_sig_all, self.X_sig_all_diff):
#                 if self.traindtype == 'diff':
#                     predict0 = model_sig.predict(Xdatadiff_sig[:2000])
#                     predict = np.concatenate([np.zeros([1,2]),np.cumsum(predict0,axis = 0)]) + self.initial
#                 else:
#                     predict = model_sig.predict(Xdata_sig[:2001])
#                 p[i].plot(predict[:,i]) 
#                 diff = np.abs(predict[:,i] - self.Ydata[:2001][:,i])
#                 error = np.max(diff)
#                 self.DIFF.append(diff)
#                 self.ERROR.append(error)
#             if self.traindtype == 'diff':
#                 predict0 = self.model.predict(self.Xdatadiff[:2000])
#                 predict = np.concatenate([np.zeros([1,2]),np.cumsum(predict0,axis = 0)]) + self.initial
#             else:
#                 predict = self.model.predict(self.Xdata[:2001])
#             p[i].plot(predict[:,i])
#             p[i].plot(self.Ydata[:2001][:,i])
#             p[i].legend(self.depth_learn +  ['Res','True'],loc = 'upper left')
#         plt.suptitle('Training path')
#         plt.savefig(name + '/2.png')
#         plt.show()
        
#         if verbose == True:
#             f,p=plt.subplots(1,2,figsize=(16,3))
#             [p[0].plot(diff) for diff in self.DIFF[:len(self.depth_learn)]]
#             p[0].legend(self.depth_learn)
#             [p[1].plot(diff) for diff in self.DIFF[len(self.depth_learn):]]
#             p[1].legend(self.depth_learn)
#             plt.show()

#             f,p=plt.subplots(1,2,figsize=(16,3))
#             p[0].plot(self.depth_learn, self.ERROR[:len(self.depth_learn)],'o-')
#             p[0].set_yscale('log')
#             p[1].plot(self.depth_learn, self.ERROR[len(self.depth_learn):],'o-')
#             p[1].set_yscale('log')
#             plt.suptitle('Training error')
#             plt.savefig(name + '/3.png')
#             plt.show()
            
#     def plot_valid(self,verbose = False):
#         generalization = self.sde.path()
#         BMpath_valid = generalization[0]

#         Xvalid = self.sde.reservoir(np.array(BMpath_valid))
#         Xvalid = np.squeeze(Xvalid)
#         Xvalid_diff = np.diff(Xvalid,axis=0)

#         Y = generalization[1]
#         Yvalid = np.squeeze(Y)

#         Xvalid_sig_all = []
#         Xvalid_sig_all_diff = []
#         for depth in self.depth_learn:
#             sig_path_stream = Sig_method.sig_stream2(np.array(BMpath_valid),depth)[0,:,:].numpy()
#             Xvalid_sig = sig_path_stream
#             Xvalid_diff_sig = np.diff(Xvalid_sig,axis=0)
#             Xvalid_sig_all.append(Xvalid_sig)
#             Xvalid_sig_all_diff.append(Xvalid_diff_sig)
            

#         self.DIFF_valid = []
#         self.ERROR_valid = []
#         f,p=plt.subplots(1,2,figsize=(16,3))
#         for i in range(2):
#             for model_sig, Xvalid_sig, Xvalid_diff_sig in zip(self.model_all, Xvalid_sig_all, Xvalid_sig_all_diff):
                
                
#                 if self.traindtype == 'diff':
#                     predict0 = model_sig.predict(Xvalid_diff_sig[:500])
#                     predict = np.concatenate([np.zeros([1,2]),np.cumsum(predict0,axis = 0)]) + self.initial
#                 else:
#                     predict = model_sig.predict(Xvalid_sig[:501])
#                 p[i].plot(predict[:,i])
#                 diff = np.abs(predict[:,i] - Yvalid[:501][:,i])
#                 error = np.max(diff)
#                 self.DIFF_valid.append(diff)
#                 self.ERROR_valid.append(error)
                
#             if self.traindtype == 'diff':
#                 predict0 = self.model.predict(Xvalid_diff[:500])
#                 predict = np.concatenate([np.zeros([1,2]),np.cumsum(predict0,axis = 0)]) + self.initial
#             else:
#                 predict = self.model.predict(Xvalid[:501])
#             p[i].plot(predict[:,i])
#             p[i].plot(Yvalid[:501][:,i])
#             p[i].legend(self.depth_learn +  ['Res','True'],loc = 'upper left')
#         plt.suptitle('Validation path')
#         plt.savefig(name + '/4.png')
#         plt.show()
        
#         if verbose == True:
#             f,p=plt.subplots(1,2,figsize=(16,3))
#             [p[0].plot(diff) for diff in self.DIFF_valid[:len(self.depth_learn)]]
#             p[0].legend(self.depth_learn)
#             [p[1].plot(diff) for diff in self.DIFF_valid[len(self.depth_learn):]]
#             p[1].legend(self.depth_learn)
#             plt.show()
        
#         f,p=plt.subplots(1,2,figsize=(16,3))
#         p[0].plot(self.depth_learn, self.ERROR_valid[:len(self.depth_learn)],'o-')
#         p[0].set_yscale('log')
#         p[1].plot(self.depth_learn, self.ERROR_valid[len(self.depth_learn):],'o-')
#         p[1].set_yscale('log')
#         plt.suptitle('Validation error')

#         plt.savefig(name + '/5.png')
#         plt.show()
        

            

    
        
        
    
        







