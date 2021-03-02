
import numpy as np
import scipy as sp
from scipy.linalg import expm
from matplotlib import pyplot as plt
import torch
torch.set_default_dtype(torch.float64)
import signatory
import Sig_method

# Linear SDE 


def get_B(label,dimensionBM,dimension):
    if label == 'diagnoal':
        B = np.zeros([dimensionBM,dimension,dimension])
        B[0,0,0] = 1
        B[0,1,1] = .5
        B[1,0,0] = 1
        B[1,1,1] = 2
    elif label == 'commute':
        P = np.array([[1,2],[3,4]])
        D1 = np.array([[1,0],[0,3]])
        D2 = np.array([[2,0],[0,4]])
        P_inv = np.linalg.inv(P)
        B1 = P@D1@P_inv
        B2 = P@D2@P_inv
        B = np.array([B1,B2])
        B = B*0.2
    elif label == 'non-commutative':
        B = np.zeros([dimensionBM,dimension,dimension])
        B[0,0,1] = 2
        B[0,1,1] = 1
        B[1,0,1] = 2
        B[1,1,1] = 0
    elif label == 'special':
        B1 = [[1,2],[3,4]]
        B2 = [[4,3],[2,1]]
        B = np.array([B1,B2])/10
    print(label)
    print(B)
    print('commutability')
    print(B[0]@B[1] - B[1]@B[0])
    return B

def semi_group_euler(state,increment,dt,B):
    dimension = state.shape[-1]
    I = np.eye(dimension)
    a = np.tensordot(increment,B,axes = 1)
    V = I + a
    dX = V@state
    return dX

class SDE_linear:
    def __init__(self,timehorizon,initialvalue,dimension,dimensionBM,timesteps,B):
        self.timehorizon = timehorizon
        self.initialvalue = initialvalue # np array
        self.dimension = dimension
        self.dimensionBM = dimensionBM
#         self.dimensionR = dimensionR
#         self.vectorfield = vectorfield
        self.timesteps = timesteps
        self.dt = timehorizon / timesteps
        self.time = np.arange(self.timesteps+1)*self.dt
        self.B = B

    def BM(self):
        BMpath_helper = np.random.normal(0,np.sqrt(self.dt),size = (self.timesteps,self.dimensionBM))
        BMpath = np.cumsum(BMpath_helper,axis = 0)
        BMpath = np.concatenate([np.zeros([1,self.dimensionBM]),BMpath],axis = 0)
        return BMpath
    
    def SDE_solver(self, initial, BMpath, name):
        if name == 'euler':
            sg = semi_group_euler
        elif name == 'analytic':
            sg = semi_group_analytic
        elif name == 'milstein':
            sg = semi_group_milstein
        SDEpath = np.zeros(shape = [self.timesteps+1, self.dimension])
        SDEpath[0,:] = initial
        for i in range(self.timesteps):
            increment = BMpath[i+1,:] - BMpath[i,:]
            SDEpath[i+1,:] = sg(SDEpath[i,:],increment,self.dt,self.B)
        return SDEpath
    
    
    
# Nonlinear SDE 

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

class SDE:
    def __init__(self,timehorizon,initialvalue,dimension,dimensionBM,vectorfield,timesteps):
        self.timehorizon = timehorizon
        self.initialvalue = initialvalue # np array
        self.dimension = dimension
        self.dimensionBM = dimensionBM
        self.vectorfield = vectorfield
        self.timesteps = timesteps
        self.dt = self.timehorizon/self.timesteps

    def path(self):
        BMpath = [np.zeros(self.dimensionBM)]
        SDEpath = [self.initialvalue]
        for i in range(self.timesteps):
            helper = np.random.normal(0,np.sqrt(self.dt),self.dimensionBM)
            BMpath = BMpath + [BMpath[-1]+helper]
            SDEpath = SDEpath + [(SDEpath[-1]+self.vectorfield(SDEpath[-1],helper))]
        self.BMpath = np.array(BMpath)
        self.SDEpath = np.array(SDEpath)
        return [self.BMpath, self.SDEpath]
    
    def plot(self):
        f1,p1=plt.subplots(1,2,figsize=(16,3))
        p1[0].plot(self.BMpath)
        p1[0].set_title('BMpath')
        p1[1].plot(self.SDEpath)
        p1[1].set_title('SDEpath')
        p1[0].grid()
        p1[1].grid()
        plt.show()
          
    
    
   