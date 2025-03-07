import numpy as np
from Optimizers import *
import wandb

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    ex = np.exp(x)
    den = np.sum(ex)
    return ex/den

class MyNeuralNetwork:
    def __init__(self, n_features, hidden_sizes, n_classes):
        '''
        Class constructor with default network hyperparameters
        '''
        self.n_features = n_features       #input size
        self.hidden_sizes = hidden_sizes   #number of neurons in hidden layers
        self.n_classes = n_classes         #output size
        
        self.weights = []                  #list of weights in each layer
        self.biases = []                   #list of biases in each layer

        self.g_activation = sigmoid
        self.o_activation = softmax

        self.batch_size = None
        self.optimizer_name = 'nag'
        self.optimizer = Fast_GD(learning_rate= 1e-3, beta = 1e-1)
    
    def params_init(self, way = "random"):
        if way == "random":
            neurons_prev = self.n_features
            for neurons_cur in self.hidden_sizes:
                self.weights.append(np.random.randn(neurons_cur,neurons_prev))
                self.biases.append(np.random.randn(neurons_cur,1))
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
        This function does a forward pass

        Arguments:
            input   : Input vector of size (n_features,1)
        
        Returns:
            out     : An output vector of probabilities of shape (n_classes,1)
            a_all   : All intermediate pre_activations
            h_all   : All intermediate activations
            
        '''
        a_all = []
        h_all = []
        h = input.reshape(-1,1)
        for i in np.arange(len(self.biases)-1):
            assert (self.weights[i]@h).reshape(-1,1).shape == self.biases[i].shape
            a = (self.weights[i]@h).reshape(-1,1)+self.biases[i]
            a_all.append(a)
            h = self.g_activation(a)
            h_all.append(h)

        assert (self.weights[-1]@h).reshape(-1,1).shape == self.biases[-1].shape
        a = self.weights[-1]@h+self.biases[-1]
        # print("Sizes of all elements in a_all")
        # for i in a_all:
        #     print(i.shape,end = " ")
        # print()
        # print("Sizes of all elements in h_all")
        # for i in h_all:
        #     print(i.shape,end = " ")
        out = self.o_activation(a)
        return out,a_all,h_all

    def compute_gradient(self, input, true_dist, y_hat, a_all, h_all,w_lookahead = None):
        '''
        This function does feed_forward and back_propagation

        Arguments:
            input   : Input vector of size (n_features,1)
            true_dist : The true distribution of the input (one hot encoded in the correct class) of size (n_classes,1)
            y_hat     : An output vector of probabilities of shape (n_classes,1)
            a_all   : All intermediate pre_activations
            h_all   : All intermediate activations
            w_lookahead : Relevant for NAG
        
        Returns:
            weight_grads : list of gradients of weights of each layer
            bias_grads   : list of gradients of biases of each layer
            
        '''
        weight_grads = []
        bias_grads = []
        true_dist = true_dist.reshape(-1,1)  #converting row to column
        assert true_dist.shape == y_hat.shape 
        cur_a_grad = -(true_dist-y_hat)  #(n_classes,1)
        delta = 1e-3 #derivative helper

        #len(self.biases) = 3
        #size of h_all is 2 starting as feed to 1st hidden layer and ending at feed to last hidden layer, final a is for output
        #w0 connects input to 1st hidden, w1 connects 1st hidden to 2nd hidden, w2 connects 2nd hidden to output
        for i in np.arange(len(self.biases))[::-1]:
            
            prev_h = h_all[i-1] if i!=0 else input.reshape(-1,1)
            weight_grads.append(cur_a_grad@(prev_h.T))  # (neurons_next,1)@(1,neurons_prev)
            bias_grads.append(cur_a_grad)                 # (neurons_next,1)
            if w_lookahead is None:
                prev_h_grad = ((self.weights[i].T)@cur_a_grad).reshape(-1,1)    # (neurons_next, neurons_prev).T@(neurons_next,1)
            else:
                assert self.weights[i].shape == w_lookahead[i].shape
                prev_h_grad = (((self.weights[i]+w_lookahead[i]).T)@cur_a_grad).reshape(-1,1)
            
            if i==0:
                break
            del_g = (self.g_activation(a_all[i-1]+delta)-self.g_activation(a_all[i-1]))/delta   # (neurons_next,1)
            prev_a_grad = prev_h_grad*del_g  #(neurons_next,1)
            cur_a_grad = prev_a_grad
        #because we come in the opposite order
        weight_grads.reverse()
        bias_grads.reverse()
        # print(f"The size of w gradients {len(weight_grads)}")
        # print(f"The size of b gradients {len(bias_grads)}")
        # for i in weight_grads:
        #     print(i.shape,end = " ")
        # print()
        return weight_grads, bias_grads

    def train(self, X_train, y_train, X_valid, y_valid, epochs):
        batch_size = X_train.shape[0] if self.batch_size is None else self.batch_size
        if self.optimizer_name == 'nag' or self.optimizer_name == 'mgd':
            for sw,sb in zip(self.weights,self.biases):
                self.optimizer.uw.append(np.zeros_like(sw))
                self.optimizer.ub.append(np.zeros_like(sb))
                
        for epoch in range(epochs):
            loss = []
            right_preds = 0
            batch_start = 0
            while batch_start <= X_train.shape[0]-1 :
                X_batch = X_train[batch_start:batch_start + batch_size,:]
                y_batch = y_train[batch_start:batch_start + batch_size,:]
                batch_start += batch_size
                dw_batch = []
                db_batch = []
                for input,true_dist in zip(X_batch,y_batch):
                    y_hat,a_all,h_all = self.feed_forward(input)
                    input = input.reshape(-1,1)
                    true_dist = true_dist.reshape(-1,1)  #changing column to row
                    if self.optimizer_name == 'nag':
                        uw = self.optimizer.uw
                        beta = self.optimizer.beta
                        for i in range(len(uw)):
                            assert uw[i].shape == self.weights[i].shape
                            uw[i] = -beta*uw[i]   #we don't use biases in gradient, so lite
                        
                        dw, db = self.compute_gradient(input,true_dist,y_hat,a_all,h_all, w_lookahead=uw)
                    else:
                        dw, db = self.compute_gradient(input,true_dist,y_hat,a_all,h_all)                   
                    if not dw_batch:
                        dw_batch = dw
                        db_batch = db
                    else:
                        assert len(dw_batch) == len(dw)
                        for i in range(len(dw_batch)):
                            assert dw_batch[i].shape == dw[i].shape
                            dw_batch[i] += dw[i]
                            assert db_batch[i].shape == db[i].shape
                            db_batch[i] += db[i]
                self.optimizer.update(self.weights, self.biases, dw_batch,db_batch)
                
            train_loss, val_loss = [], []
            train_right, val_right = 0,0
            for input,true_dist in zip(X_train, y_train):
                y_hat,_,_ = self.feed_forward(input)
                true_dist = true_dist.reshape(-1,1)
                train_loss.append(np.sum(np.square(true_dist - y_hat)))
                train_right += (np.argmax(true_dist)==np.argmax(y_hat))

            for input,true_dist in zip(X_valid, y_valid):
                y_hat,_,_ = self.feed_forward(input)
                true_dist = true_dist.reshape(-1,1)
                val_loss.append(np.sum(np.square(true_dist - y_hat)))
                val_right += (np.argmax(true_dist)==np.argmax(y_hat))
            
            train_acc = train_right*(1.0)/y_train.shape[0]
            val_acc = val_right*(1.0)/y_valid.shape[0]

            wandb.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": np.mean(train_loss),
                    "val_acc": val_acc,
                    "val_loss": np.mean(val_loss),
                }
            )

            if epoch % int(epochs/10) == 0:
                print(f"Epoch {epoch}, T_Loss: {np.mean(train_loss)}, T_acc: {train_acc}",end = ', ')
                print(f"V_Loss: {np.mean(val_loss)}, V_acc: {val_acc}")
        try:
            self.optimizer.clear_history()
        except Exception as e:
            pass            

    def predict(self, X):
        y = []
        for input in X:
            y.append(self.feed_forward(input))
        return np.stack(y)
