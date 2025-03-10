import numpy as np


class GD:
    def __init__(self,learning_rate = 1e-3,alpha = 0.1):
        self.learning_rate = learning_rate
        self.alpha = alpha
        
    def update(self, weights, biases, weight_gradients, bias_gradients):
        for i in range(len(weights)):
            assert np.sum(np.isnan(weight_gradients[i])) == 0, f"{i}"
            assert np.sum(np.isnan(bias_gradients[i])) == 0, f"{i}"
            weights[i] -= self.learning_rate*weight_gradients[i] + self.learning_rate*self.alpha*weights[i]
            biases[i] -= self.learning_rate*bias_gradients[i] 

    def clear_history(self):
        pass

class Fast_GD:
    def __init__(self,learning_rate = 1e-3,beta  =0.9,alpha = 0.1):
        self.learning_rate = learning_rate
        self.beta = beta
        self.alpha = alpha
        self.uw = []
        self.ub = []

    def update(self, weights, biases, weight_gradients, bias_gradients):

        for i in range(len(weights)):
            # print(len(self.uw))
            # print(len(weight_gradients))
            # print(len(self.ub))
            # print(len(bias_gradients))
            if len(self.uw)!=len(weight_gradients):
                self.uw.append(weight_gradients[i])
                self.ub.append(bias_gradients[i])
            else:
                self.uw[i] = self.beta*self.uw[i] + weight_gradients[i]  
                self.ub[i] = self.beta*self.ub[i] + bias_gradients[i]    

            weights[i] -= self.learning_rate*self.uw[i] + self.learning_rate*self.alpha*weights[i]
            biases[i] -= self.learning_rate*self.ub[i] 
    def clear_history(self):
        self.uw = []
        self.ub = []

class RMSprop:
    def __init__(self,learning_rate = 1e-3,beta = 0.9,alpha = 0.1,epsilon = 1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.alpha = alpha
        self.vw, self.vb = [], []
        self.epsilon = epsilon

    def update(self, weights, biases, weight_gradients, bias_gradients):
        for i in range(len(weights)):
            if len(self.vw)!=len(weight_gradients):
                self.vw.append((1-self.beta)*(weight_gradients[i]**2))
                self.vb.append((1-self.beta)*(bias_gradients[i]**2))
            else:
                self.vw[i] = self.beta*self.vw[i] + (1-self.beta)*(weight_gradients[i]**2)  
                self.vb[i] = self.beta*self.vb[i] + (1-self.beta)*(bias_gradients[i]**2)    

            weights[i] -= self.learning_rate*weight_gradients[i]/np.sqrt(self.vw[i]+self.epsilon) + self.learning_rate*self.alpha*weights[i]
            biases[i] -= self.learning_rate*bias_gradients[i]/np.sqrt(self.vb[i]+self.epsilon) 

    def clear_history(self):
        self.vw = []
        self.vb = []

class Adam:
    def __init__(self,learning_rate = 1e-3,beta_m = 0.9,beta_v = 0.99,alpha = 0.1,epsilon = 1e-6):
        self.learning_rate = learning_rate
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.alpha = alpha
        self.time = 1
        self.mw, self.mb =[], []
        self.vw, self.vb = [], []
        self.epsilon = epsilon

    def update(self, weights, biases, weight_gradients, bias_gradients):
        for i in range(len(weights)):

            if len(self.vw)!=len(weight_gradients):
                # print("Init done")
                self.mw.append((1-self.beta_m)*(weight_gradients[i]))
                self.mb.append((1-self.beta_m)*(bias_gradients[i]))
                self.vw.append((1-self.beta_v)*(weight_gradients[i]**2))
                self.vb.append((1-self.beta_v)*(bias_gradients[i]**2))
            else:
                self.mw[i] = self.beta_m*self.mw[i] + (1-self.beta_m)*(weight_gradients[i])  
                self.mb[i] = self.beta_m*self.mb[i] + (1-self.beta_m)*(bias_gradients[i])   
                self.vw[i] = self.beta_v*self.vw[i] + (1-self.beta_v)*(weight_gradients[i]**2)  
                self.vb[i] = self.beta_v*self.vb[i] + (1-self.beta_v)*(bias_gradients[i]**2)    

            mw_bias_corrected, mb_bias_corrected = self.mw[i]/(1-pow(self.beta_m,self.time)), self.mb[i]/(1-pow(self.beta_m,self.time))
            vw_bias_corrected, vb_bias_corrected = self.vw[i]/(1-pow(self.beta_v,self.time)), self.vb[i]/(1-pow(self.beta_v,self.time)) 

            weights[i] -= self.learning_rate*mw_bias_corrected/(np.sqrt(vw_bias_corrected)+self.epsilon) + self.learning_rate*self.alpha*weights[i]
            biases[i] -= self.learning_rate*mb_bias_corrected/(np.sqrt(vb_bias_corrected)+self.epsilon) 

    def clear_history(self):
        self.mw = []
        self.mb = []
        self.vw = []
        self.vb = []
        self.time +=1

class NAdam:
    def __init__(self,learning_rate = 1e-3,beta_m = 0.9,beta_v = 0.99,alpha = 0.1,epsilon = 1e-6):
        self.learning_rate = learning_rate
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.alpha = alpha
        self.time = 1
        self.mw, self.mb =[], []
        self.vw, self.vb = [], []
        self.epsilon = epsilon

    def update(self, weights, biases, weight_gradients, bias_gradients):
        for i in range(len(weights)):
            if len(self.vw)!=len(weight_gradients):
                self.mw.append((1-self.beta_m)*(weight_gradients[i]))
                self.mb.append((1-self.beta_m)*(bias_gradients[i]))
                self.vw.append((1-self.beta_v)*(weight_gradients[i]**2))
                self.vb.append((1-self.beta_v)*(bias_gradients[i]**2))
            else:
                self.mw[i] = self.beta_m*self.mw[i] + (1-self.beta_m)*(weight_gradients[i])  
                self.mb[i] = self.beta_m*self.mb[i] + (1-self.beta_m)*(bias_gradients[i])   
                self.vw[i] = self.beta_v*self.vw[i] + (1-self.beta_v)*(weight_gradients[i]**2)  
                self.vb[i] = self.beta_v*self.vb[i] + (1-self.beta_v)*(bias_gradients[i]**2)    

            mw_bias_corrected, mb_bias_corrected = self.mw[i]/(1-pow(self.beta_m,self.time)), self.mb[i]/(1-pow(self.beta_m,self.time))
            vw_bias_corrected, vb_bias_corrected = self.vw[i]/(1-pow(self.beta_v,self.time)), self.vb[i]/(1-pow(self.beta_v,self.time)) 

            w_lookahead = weight_gradients[i]/((1-pow(self.beta_m,self.time+1)))
            b_lookahead = bias_gradients[i]/((1-pow(self.beta_m,self.time+1)))

            w_update = (1-self.beta_m)*w_lookahead + self.beta_m*mw_bias_corrected
            b_update = (1-self.beta_m)*b_lookahead + self.beta_m*mb_bias_corrected

            weights[i] -= self.learning_rate*w_update/(np.sqrt(vw_bias_corrected)+self.epsilon) + self.learning_rate*self.alpha*weights[i]
            biases[i] -= self.learning_rate*b_update/(np.sqrt(vb_bias_corrected)+self.epsilon) 

    def clear_history(self):
        self.mw = []
        self.mb = []
        self.vw = []
        self.vb = []
        self.time += 1

OPTIMIZER_MAP = {
    'sgd': GD,
    'momentum': Fast_GD,
    'nag': Fast_GD,
    'rmsprop': RMSprop,
    'adam': Adam,
    'nadam': NAdam
}