import numpy as np


class GD:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
        
    def update(self, weights, biases, weight_gradients, bias_gradients):
        for i in range(len(weights)):
            assert np.sum(np.isnan(weight_gradients[i])) == 0, f"{i}"
            assert np.sum(np.isnan(bias_gradients[i])) == 0, f"{i}"
            weights[i] -= self.learning_rate*weight_gradients[i]
            biases[i] -= self.learning_rate*bias_gradients[i]

class Fast_GD:
    def __init__(self,learning_rate,beta):
        self.learning_rate = learning_rate
        self.beta = beta
        self.uw = []
        self.ub = []

    def update(self, weights, biases, weight_gradients, bias_gradients):
        for i in range(len(weights)):
            self.uw[i] = self.beta*self.uw[i] + weight_gradients[i]  if not self.uw  else weight_gradients[i]
            weights[i] -= self.learning_rate*self.uw[i]
            self.ub[i] = self.beta*self.ub[i] + bias_gradients[i]    if not self.ub  else bias_gradients[i]
            biases[i] -= self.learning_rate*self.ub[i]
    def clear_history(self):
        self.uw = []
        self.ub = []

OPTIMIZER_MAP = {
    'gd': GD,
    'mgd': Fast_GD,
    'nag': Fast_GD 
}