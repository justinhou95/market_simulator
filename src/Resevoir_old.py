import numpy as np
import Sig_method
import sklearn
from sklearn import linear_model
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

name = 'data'

d=2
M=150

def nilpotent(M):
    B = np.zeros((M,M))
    for i in range(2,M):
        B[i,i-1]=1.0
    return B

def canonical(i,M):
    e = np.zeros((M,1))
    e[i,0]=1.0
    return e

def vectorfieldoperator(state,increment):
    d = np.shape(increment)[0]
    N = np.shape(state)[0]
    direction = np.zeros((N,1))
    for i in range(d):
        helper = np.zeros((N,1))
        for j in range(N):
            helper[j]=np.sin((j+1)*state[j,0])
        direction=direction + helper*increment[i]
    return direction

def vectorfield2dsimple(state,increment):
    return np.array([state[0],state[1]])*increment[0]\
            +np.array([state[0],state[1]])*increment[1]

def vectorfield2dlinear(state,increment):
    return np.array([2.0*state[1],1.0*state[1]])*increment[0]\
            +np.array([2.0*state[1],0.0*state[1]])*increment[1]

def vectorfield2d(state,increment):
    return np.array([(2.0*np.sqrt(state[1]**2))**0.7,1.0*state[1]])*increment[0]\
            +np.array([(2.0*np.sqrt(state[1]**2))**0.7,0.0*state[1]])*increment[1]

def vectorfield3d(state,increment):
    return np.array([np.sin(5*state[0])*np.exp(-state[2]),np.cos(5*state[1]),-state[2]*state[1]])*increment[0]+np.array([np.sin(4*state[1]),np.cos(4*state[0]),-state[0]*state[1]])*increment[1]
def vectorfield(state,increment):
    return 5*np.exp(-state)*increment[0] + 5*np.cos(state)*increment[1]
def randomAbeta(d,M):
    A = []
    beta = []
    for i in range(d):
        B = 0.0*nilpotent(M) + np.random.normal(0.0,0.5,size=(M,M)) 
        B = np.random.permutation(B)
        A = A + [B]
        beta = beta + [0.0*canonical(i,M)+np.random.normal(0.0,0.5,size=(M,1))]
    return [A,beta]

Abeta = randomAbeta(d,M)
A = Abeta[0]
beta = Abeta[1]

def sigmoid(x):
    return np.tanh(x)

def reservoirfield(state,increment):
    value = np.zeros((M,1))
    for i in range(d):
        value = value + sigmoid(np.matmul(A[i],state) + beta[i])*increment[i]
    return value

def cut_path(path,sublength):
    start = 0
    end = start + sublength 
    path_split = []
    while start < path.shape[0]-5:
        path_split.append(path[start:end])
        start += sublength
        end = start + sublength
    return np.array(path_split)


class SDE:
    def __init__(self,timehorizon,initialvalue,dimension,dimensionBM,dimensionR,vectorfield,timesteps):
        self.timehorizon = timehorizon
        self.initialvalue = initialvalue # np array
        self.dimension = dimension
        self.dimensionBM = dimensionBM
        self.dimensionR = dimensionR
        self.vectorfield = vectorfield
        self.timesteps = timesteps

    def path(self):
        BMpath = [np.zeros(self.dimensionBM)]
        SDEpath = [self.initialvalue]
        for i in range(self.timesteps):
            helper = np.random.normal(0,np.sqrt(self.timehorizon/self.timesteps),self.dimensionBM)
            BMpath = BMpath + [BMpath[-1]+helper]
            SDEpath = SDEpath + [np.exp(-0.0*self.timehorizon/self.timesteps)*(SDEpath[-1]+self.vectorfield(SDEpath[-1],helper))]

        return [BMpath, SDEpath]
    
    def anypath(self):
        BMpath = [np.zeros(self.dimensionBM)]
        SDEpath = [np.array([1.0, self.initialvalue])]#[np.ones((self.dimension,1))*self.initialvalue]
        
        for i in range(self.timesteps):
            helper = np.cos(BMpath[-1]*50)*self.timehorizon/self.timesteps#np.random.normal(0,np.sqrt(self.timehorizon/self.timesteps),self.dimensionBM)
            BMpath = BMpath + [BMpath[-1]+helper]
            SDEpath = SDEpath + [np.exp(-0.0*self.timehorizon/self.timesteps)*(SDEpath[-1]+self.vectorfield(SDEpath[-1],helper))]
            
        return [BMpath, SDEpath]
        
    def reservoir(self,BMpath, flag = 'initial'):
        helper = canonical(0,self.dimensionR)
        if flag != 'naive':
            helper[:self.dimension,0] = self.initialvalue
        reservoirpath = [helper]
        l_helper = BMpath.shape[0]
        for i in range(l_helper-1):
            increment = BMpath[i+1]-BMpath[i]
            reservoirpath = reservoirpath + [np.exp(-0.0*self.timehorizon/self.timesteps)*(reservoirpath[-1]+reservoirfield(reservoirpath[-1],increment))]
        return reservoirpath    
    
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
    def __init__(self, sde, training):
        self.sde = sde
        self.training = training
        self.BMpath=np.array(self.training[0])
        self.SDEpath=np.array(self.training[1])  
        self.initial = self.SDEpath[0,:]
        
    def prepare(self, depth_learn):
        self.depth_learn = depth_learn
        self.Ydata = self.SDEpath
        print(self.Ydata.shape,end=', ')
        
        
        X = self.sde.reservoir(np.array(self.BMpath),'naive')
        self.X = np.squeeze(X)
        self.Xdata0 = np.tensordot(self.X,self.initial,axes = 0)
        self.Xdata = np.reshape(self.Xdata0,[-1,self.Xdata0.shape[-1]*self.Xdata0.shape[-2]])  
        print(self.Xdata.shape,end=', ')
        
        
        self.X_sig_all = []
        for depth in self.depth_learn:
            sig_path_stream = Sig_method.sig_stream2(np.array(self.BMpath),depth)[0,:,:].numpy()
            Xdata_sig = sig_path_stream
            self.X_sig_all.append(Xdata_sig)
            print(Xdata_sig.shape,end=', ')
        print('')

    def prepare_split(self, depth_learn, sublength, trainto):
        self.sublength = sublength
        self.BMpath_split = cut_path(self.BMpath[:trainto+1],sublength)    
        # (batch, sublength, dimensionBM)   i.e. (2,500,2)  non-intersection between different subpath
        
        self.SDEpath_split = cut_path(self.SDEpath[:trainto+1],sublength)    # (batch, sublength, dimension)
        self.initial_split = np.array([path[0,:] for path in self.SDEpath_split]) # (batch, dimension)
        self.depth_learn = depth_learn  # depth > sublength include no further information but increase the dimension
        
        self.Ytrain0 = self.SDEpath_split # (batch, sublength, dimension)
        self.Ytrain = np.reshape(self.Ytrain0,[-1,self.Ytrain0.shape[-1]]) # (batch*sublength, dimension)
        print(self.Ytrain.shape,end=', ')
        
        self.resevior_path_split = [np.reshape(np.array(self.sde.reservoir(p,'naive')),[-1,self.sde.dimensionR]) for p in self.BMpath_split]
        # (batch, sublength, Resevior_dimension)
        
        self.Xtrain0 = np.array([np.tensordot(x,y,axes = 0) \
                                for x,y in zip(self.resevior_path_split, self.initial_split)])      
        # (batch, sublength, Res_dim, dimension)
        
        self.Xtrain = np.reshape(self.Xtrain0,[-1,self.Xtrain0.shape[-1]*self.Xtrain0.shape[-2]])  
        self.Xtrain_nonlinear = np.reshape(self.Xtrain0,[-1,self.Xtrain0.shape[-2], self.Xtrain0.shape[-1],1])  
        # (batch*sublength, Res_dim*dimension)
        
        self.Xtrain10 = np.array([np.tile(x,[self.sublength,1]) for x in self.initial_split])
        self.Xtrain1 = np.reshape(self.Xtrain10,[-1,self.sde.dimension])   
        
        print(self.Xtrain.shape,end=', ')
        
        
        self.X_sig_train_all = []
        for depth in self.depth_learn:
            self.sig_path_stream_split = [Sig_method.sig_stream2(np.array(path),depth)[0,:,:].numpy() for path in self.BMpath_split]
            # (batch, sublength+1, dim_sig)
            # Note that the first sig is always (1,0,...,0)
            Xtrain_sig0 = np.array([np.tensordot(x,y,axes = 0) \
                                for x,y in zip(self.sig_path_stream_split, self.initial_split)])
            # (batch, sublength+1, dim_sig * dimension)
            # Note that Xtrain[:][0,:dimension]  == initial_split[:]
            Xtrain_sig = np.reshape(Xtrain_sig0,[-1,Xtrain_sig0.shape[-1]*Xtrain_sig0.shape[-2]]) 
            self.X_sig_train_all.append(Xtrain_sig)
            print(np.shape(Xtrain_sig),end=', ')
        print('')
            
    
    def train_split(self, alpha, verbose = False, fix_intercept = False):
        self.lm = linear_model.Ridge(alpha = alpha, fit_intercept = fix_intercept)   
        self.model = self.lm.fit(self.Xtrain,self.Ytrain)
        if verbose:
            print('score: ',self.model.score(self.Xtrain,self.Ytrain))
            print('max coefficient: ',np.max(np.abs(self.model.coef_)))
        self.lm_all = []
        self.model_all = []
        for Xtrain_sig in self.X_sig_train_all:
            lm_sig = linear_model.Ridge(alpha = alpha, fit_intercept = fix_intercept)
            model_sig = lm_sig.fit(Xtrain_sig,self.Ytrain)
            self.lm_all.append(lm_sig)
            self.model_all.append(model_sig)
            if verbose:
                print('score: ',model_sig.score(Xtrain_sig,self.Ytrain))
                print('max coefficient: ',np.max(np.abs(model_sig.coef_)))
                print('alpha： ',lm_sig.alpha_)
    
    def plot_train_split(self,verbose = False):
        
        f,p=plt.subplots(1,2,figsize=(16,3))
        self.DIFF = []
        self.ERROR = []
        for i in range(2):
            for model_sig, Xdata_sig in zip(self.model_all, self.X_sig_all):
                helper0 = np.tensordot(Xdata_sig, self.initial, axes = 0)
                helper = np.reshape(helper0,[-1,helper0.shape[-1]*helper0.shape[-2]])
                predict = model_sig.predict(helper)

                p[i].plot(predict[:,i]) 
                diff = np.abs(predict[:,i] - self.Ydata[:][:,i])
                error = np.max(diff)
                self.DIFF.append(diff)
                self.ERROR.append(error)
            predict = self.model.predict(self.Xdata[:])
            p[i].plot(predict[:,i])
            p[i].plot(self.Ydata[:][:,i])
            p[i].legend(self.depth_learn +  ['Res','True'],loc = 'upper left')
        plt.suptitle('Training path')
        plt.savefig(name + '/2.png')
        plt.show()
        
        if verbose == True:
            f,p=plt.subplots(1,2,figsize=(16,3))
            [p[0].plot(diff) for diff in self.DIFF[:len(self.depth_learn)]]
            p[0].legend(self.depth_learn)
            [p[1].plot(diff) for diff in self.DIFF[len(self.depth_learn):]]
            p[1].legend(self.depth_learn)
            plt.show()

            f,p=plt.subplots(1,2,figsize=(16,3))
            p[0].plot(self.depth_learn, self.ERROR[:len(self.depth_learn)],'o-')
            p[0].set_yscale('log')
            p[1].plot(self.depth_learn, self.ERROR[len(self.depth_learn):],'o-')
            p[1].set_yscale('log')
            plt.suptitle('Training error')
            plt.savefig(name + '/3.png')
            plt.show()
            
            
    def valid_split(self,validto, depth_learn):
        self.depth_learn = depth_learn
        self.generalization = self.sde.path()
        self.BMpath_valid = self.generalization[0][0:validto+1]
        
        Y = self.generalization[1][0:validto+1]
        self.Yvalid = np.squeeze(Y)

        Xvalid0 = self.sde.reservoir(np.array(self.BMpath_valid),'naive')
        self.Xvalid0 = np.squeeze(Xvalid0)
        self.Xvalid0 = np.tensordot(self.Xvalid0,self.initial,axes = 0)
        self.Xvalid = np.reshape(self.Xvalid0,[-1,self.Xvalid0.shape[-1]*self.Xvalid0.shape[-2]])  

        self.Xvalid_sig_all = []
        for depth in self.depth_learn:
            sig_path_stream = Sig_method.sig_stream2(np.array(self.BMpath_valid),depth)[0,:,:].numpy()
            Xdata_sig = sig_path_stream
            self.Xvalid_sig_all.append(Xdata_sig)
        
        return self.generalization, self.BMpath_valid, self.Yvalid, self.Xvalid, self.Xvalid_sig_all
    
    def valid_split_set(self,validto, depth_learn, set_size = 1):
        self.generalization_set = []
        self.BMpath_valid_set = []
        self.Yvalid_set = []
        self.Xvalid_set = []
        self.Xvalid_sig_all_set = []
        for i in tqdm(range(set_size)):
            generalization, BMpath_valid, Yvalid, Xvalid, Xvalid_sig_all = \
            self.valid_split(validto, depth_learn)
            self.generalization_set.append(generalization)
            self.BMpath_valid_set.append(BMpath_valid)
            self.Yvalid_set.append(Yvalid)
            self.Xvalid_set.append(Xvalid)
            self.Xvalid_sig_all_set.append(Xvalid_sig_all)
            
    def validation_in_set(self,validto):
        self.res_predict_set = []
        for Xvalid, Yvalid in zip(self.Xvalid_set,self.Yvalid_set):
            res_predict = self.model.predict(Xvalid[:validto + 1])
            self.res_predict_set.append(res_predict)
        #     plt.plot(predict)
        #     plt.plot(Yvalid[:validto + 1])
        #     plt.show()
        
        self.sig_predict_set = []
        for Xvalid_sig_all, Yvalid in zip(self.Xvalid_sig_all_set,self.Yvalid_set):
            sig_predict = []
            for model_sig, Xvalid_sig in zip(self.model_all, Xvalid_sig_all):
                helper0 = np.tensordot(Xvalid_sig[:validto + 1], self.initial, axes = 0)
                helper = np.reshape(helper0,[-1,helper0.shape[-1]*helper0.shape[-2]])
                predict = model_sig.predict(helper)
                sig_predict.append(predict)
        #         plt.plot(predict)
        #         plt.plot(Yvalid[:validto + 1])
        #         plt.show()
            self.sig_predict_set.append(sig_predict) 
        self.X_res = tf.convert_to_tensor(self.res_predict_set)
        self.X_sig0 = tf.convert_to_tensor(self.sig_predict_set)
        self.X_sig = []
        for i in range(len(self.depth_learn)):
            self.X_sig.append(self.X_sig0[:,i,:,:])
        
    def plot_valid_split(self,validto, verbose = False):
        self.DIFF_valid = []
        self.ERROR_valid = []
        f,p=plt.subplots(1,2,figsize=(16,3))
        for i in range(2):
            for model_sig, Xvalid_sig in zip(self.model_all, self.Xvalid_sig_all):
                helper0 = np.tensordot(Xvalid_sig[:validto + 1], self.initial, axes = 0)
                helper = np.reshape(helper0,[-1,helper0.shape[-1]*helper0.shape[-2]])
                predict = model_sig.predict(helper)
                p[i].plot(predict[:,i])
                diff = np.abs(predict[:,i] - self.Yvalid[:validto + 1][:,i])
                error = np.max(diff)
                self.DIFF_valid.append(diff)
                self.ERROR_valid.append(error)
                
            predict = self.model.predict(self.Xvalid[:validto + 1])
            p[i].plot(predict[:,i])
            p[i].plot(self.Yvalid[:validto + 1][:,i])
            p[i].legend(self.depth_learn +  ['Res','True'],loc = 'upper left')
        plt.suptitle('Validation path')
        plt.savefig(name + '/4.png')
        plt.show()
        
        if verbose == True:
            f,p=plt.subplots(1,2,figsize=(16,3))
            [p[0].plot(diff) for diff in self.DIFF_valid[:len(self.depth_learn)]]
            p[0].legend(self.depth_learn)
            [p[1].plot(diff) for diff in self.DIFF_valid[len(self.depth_learn):]]
            p[1].legend(self.depth_learn)
            plt.show()
        
        f,p=plt.subplots(1,2,figsize=(16,3))
        p[0].plot(self.depth_learn, self.ERROR_valid[:len(self.depth_learn)],'o-')
        p[0].set_yscale('log')
        p[1].plot(self.depth_learn, self.ERROR_valid[len(self.depth_learn):],'o-')
        p[1].set_yscale('log')
        plt.suptitle('Validation error')

        plt.savefig(name + '/5.png')
        plt.show()
        
        
        
        
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
        

            

    
        
        
    
        



