import numpy as np
from Optimizers import *
from Activations import *
import wandb


class MyNeuralNetwork:
    def __init__(self, n_features, hidden_sizes, n_classes):
        '''
        Class constructor with default network hyperparameters
        '''
        self.loss = 'cross_entropy'
        self.n_features = n_features       #input size
        self.hidden_sizes = hidden_sizes   #number of neurons in hidden layers
        self.n_classes = n_classes         #output size
        
        self.weights = []                  #list of weights in each layer
        self.biases = []                   #list of biases in each layer

        self.g_activation = sigmoid
        self.o_activation = softmax

        self.batch_size = None
        self.optimizer_name = 'sgd'
        self.optimizer = GD
    
    def params_init(self,weights = None, biases = None, way = "random", seed = 42):
        '''
        Function to initialise weights and biases
        '''
        r = np.random.RandomState(seed)
        if way == "random":
            neurons_prev = self.n_features
            for neurons_cur in self.hidden_sizes:
                self.weights.append(r.randn(neurons_cur,neurons_prev).astype(np.float32)*0.05)
                self.biases.append(np.zeros((neurons_cur,1)).astype(np.float32))
                neurons_prev = neurons_cur
        elif way == "Xavier":
            #normal xavier
            neurons_prev = self.n_features
            for neurons_cur in self.hidden_sizes:
                sigma = np.sqrt(2/neurons_prev) if self.g_activation == 'ReLU' else np.sqrt(1/neurons_prev)
                self.weights.append(r.randn(neurons_cur,neurons_prev).astype(np.float32)*sigma)
                self.biases.append(np.zeros((neurons_cur,1)).astype(np.float32))
                neurons_prev = neurons_cur
        
        elif way =="Custom":
            neurons_prev = self.n_features
            for i,neurons_cur in enumerate(self.hidden_sizes):
                self.weights.append(weights[i])
                self.biases.append(biases[i])
                neurons_prev = neurons_cur


    def update_network_hp(self,**kwargs):
        '''
        Function to update network hyperparameters
        '''
        for key, value in kwargs.items():
            if value is not None and hasattr(self,key):
                setattr(self,key,value)

    def update_optim_hp(self,optimizer_name, **kwargs):
        '''
        Function to update optimizer hyperparameters
        '''
        self.optimizer_name = optimizer_name
        self.optimizer = OPTIMIZER_MAP[optimizer_name](**kwargs)

    def feed_forward(self,input):
        '''
        This function does a forward pass and returns the output probabilities, activations and pre_activations for a data point 
        '''
        a_all = []
        h_all = []
        h = input.T  #(n_features, n_samples) 
        for i in np.arange(len(self.biases)-1):
            a = self.weights[i]@h+self.biases[i].reshape(-1,1)  #(n_cur,n_prev)@(n_prev,n_samples) + (n_cur,1)
            a_all.append(a)   #(n_cur,n_samples)
            h = self.g_activation(a) 
            assert h.shape == a.shape
            h_all.append(h) #(n_cur,n_samples)

        a = self.weights[len(self.weights)-1]@h+self.biases[len(self.biases)-1].reshape(-1,1)   #(n_classes, n_samples)
        out = self.o_activation(a).T  #(n_classes, n_samples).T
        assert out.shape == (input.shape[0],self.n_classes)
        return out,a_all,h_all

    def backprop(self, input, true_dist, y_hat, a_all, h_all,w_lookahead = None):
        '''
        This function does back_propagation and returns the weight and bias gradients for a data point           
        '''
        weight_grads = []
        bias_grads = []
        true_dist = true_dist.T  #(n_classes, n_samples)
        y_hat = y_hat.T          #(n_classes, n_samples)
        assert true_dist.shape == y_hat.shape 
        if self.loss == 'cross_entropy':
            cur_a_grad = -(true_dist-y_hat)  #(n_classes,n_samples)
        elif self.loss == 'mean_squared_error':
            cur_a_grad = -(true_dist-y_hat)*(y_hat)*(1-y_hat)  #(n_classes,n_samples)

        for i in np.arange(len(self.biases))[::-1]:
            
            prev_h = h_all[i-1] if i!=0 else input.T   #(n_prev, n_samples)
            weight_grads.append(cur_a_grad@(prev_h.T))  # (n_next,n_samples)@(n_samples,neurons_prev)
            bias_grads.append(np.sum(cur_a_grad,axis = 1).reshape(-1,1))                 # (n_next,1)

            if w_lookahead is None:
                prev_h_grad = ((self.weights[i].T)@cur_a_grad)    # (n_next, n_prev).T@(n_next,n_samples) = 
            else:
                #Handling lookahead cases
                assert self.weights[i].shape == w_lookahead[i].shape
                prev_h_grad = (((self.weights[i]+w_lookahead[i]).T)@cur_a_grad)  # (n_prev, n_samples)
            
            if i==0:
                break

            del_g = D_ACTIVATIONS_MAP[self.g_activation](a_all[i-1])   # (n_prev,n_samples)
            prev_a_grad = prev_h_grad*del_g  #(n_prev,n_samples)
            cur_a_grad = prev_a_grad

        weight_grads.reverse()
        bias_grads.reverse()

        for i in range(len(self.weights)):
            assert weight_grads[i].shape == self.weights[i].shape
            assert bias_grads[i].shape == self.biases[i].shape
        return weight_grads, bias_grads
    
    def print_metrics(self, X_train, y_train, X_valid, y_valid, epoch):
        '''
        This function prints the metrics for predictions at any instance
        '''
        epsilon =1e-8
        
        y_hat_t,_,_ = self.feed_forward(X_train)  #(n_samples, n_classes)
        y_hat_v,_,_ = self.feed_forward(X_valid)

        if self.loss == 'cross_entropy':
            train_loss = -np.mean(np.sum(y_train*np.log(y_hat_t+epsilon),axis = 1))
            valid_loss = -np.mean(np.sum(y_valid*np.log(y_hat_v+epsilon),axis = 1))
        elif self.loss == 'mean_squared_error':
            train_loss = np.mean(np.sum(np.square(y_train-y_hat_t),axis = 1))
            valid_loss = np.mean(np.sum(np.square(y_valid-y_hat_v),axis = 1))   

            
        train_acc = np.mean(np.argmax(y_train,axis = 1) == np.argmax(y_hat_t,axis = 1))
        valid_acc = np.mean(np.argmax(y_valid,axis = 1) == np.argmax(y_hat_v,axis = 1))
        print(f"Epoch {epoch}, T_Loss: {np.mean(train_loss)}, T_acc: {train_acc}",end = ', ')
        print(f"V_Loss: {np.mean(valid_loss)}, V_acc: {valid_acc}")  

        try:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": valid_acc,
                    "val_loss": valid_loss,
                }
            )
        except Exception:
            pass



    def train(self, X_train, y_train, X_valid, y_valid, epochs = 30):
        '''
        This function trains a neural network and learns the parameters
        '''
        batch_size = X_train.shape[0] if self.batch_size is None else self.batch_size      
        self.print_metrics(X_train, y_train, X_valid, y_valid, 0, log = False)      
        for epoch in range(1,epochs+1):
            batch_start = 0
            while batch_start <= X_train.shape[0]-1 :
                X_batch = X_train[batch_start:batch_start + batch_size,:]
                y_batch = y_train[batch_start:batch_start + batch_size,:]
                batch_start += batch_size

                y_hat, a_all, h_all = self.feed_forward(X_batch)
                
                #Handling Nesterov to create lookahead step
                if self.optimizer_name == 'nag':
                    uw = self.optimizer.uw
                    beta = self.optimizer.beta
                    wla = []
                    for i in range(len(self.weights)):
                        wla.append(np.zeros_like(self.weights[i]) if not uw else -beta*uw[i]) 
                    dw_batch, db_batch = self.backprop(X_batch,y_batch,y_hat,a_all,h_all, w_lookahead=wla)
                else:
                    dw_batch, db_batch = self.backprop(X_batch,y_batch,y_hat,a_all,h_all)                   
                assert len(dw_batch) == len(self.weights)
                assert len(db_batch) == len(self.biases)

                self.optimizer.update(self.weights, self.biases, dw_batch,db_batch)   

            self.print_metrics(X_train, y_train, X_valid, y_valid, epoch, log = True)      

    def predict(self, X):
        '''
        This function returns the prediction given any data vector
        '''
        y_hat,_,_ = self.feed_forward(X)
        return y_hat

    def clear_opt_history(self):
        '''
        This function clears the history of gradients stored in the optimizer
        '''
        self.optimizer.clear_history()   