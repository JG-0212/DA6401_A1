import numpy as np

def sigmoid(x):
    x = np.clip(x,-500,500)
    return np.exp(x)/(1+np.exp(x)) 

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    x = np.clip(x,-250,250)
    return (np.exp(2*x)-1)/(np.exp(2*x)+1) 

def d_tanh(x):
    return 1-tanh(x)**2

def ReLU(x):
    return np.maximum(0,x)

def d_ReLU(x):
    return np.where(x>0,1,0)

def identity(x):
    return x

def d_identity(x):
    return np.ones_like(x)

def softmax(x):
    ex = np.exp(x-np.max(x,axis = 0))
    den = np.sum(ex, axis = 0)
    return ex/den

ACTIVATIONS_MAP = {
    'tanh': tanh,
    'sigmoid': sigmoid,
    'ReLU': ReLU,
    'identity': identity,
    'softmax': softmax
}

D_ACTIVATIONS_MAP = {
    tanh: d_tanh,
    sigmoid: d_sigmoid,
    ReLU: d_ReLU,
    identity: d_identity
}