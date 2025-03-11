from Data import *
from NeuralNetwork import *
from Optimizers import *
from Activations import *
import yaml
import wandb
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #Loading and processing the data
    (train_images, train_labels), (test_images, test_labels)   =  load_data(name = 'fashion_mnist')
    train_images, train_labels, valid_images, valid_labels = split_validation(train_images, train_labels, valid_size=0.1, seed = 42)
    train_images, train_labels,_,_ = split_validation(train_images, train_labels, valid_size=0.95, seed = 42)
    X_train, X_valid, X_test = preprocess(train_images), preprocess(valid_images), preprocess(test_images)
    y_train, y_valid, y_test = one_hot_encoded(train_labels), one_hot_encoded(valid_labels), one_hot_encoded(test_labels)

    #Reading the config file
    with open("./config_fine.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    run = wandb.init(config=config)

    #Getting the configutation from wandb sweep
    num_layers = wandb.config.num_layers
    hidden_size = wandb.config.hidden_size
    lr = wandb.config.learning_rate
    beta = wandb.config.beta
    beta_m = wandb.config.beta1
    beta_v = wandb.config.beta2
    alpha = wandb.config.weight_decay
    winit = wandb.config.weight_init
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs
    optimizer_name = wandb.config.optimizer
    g_activation = wandb.config.activation

    #Renaming our run
    run.name = f"nhl:{num_layers}, sz: {hidden_size}, lr:{lr}, beta: {beta}, beta1:{beta_m}, beta2:{beta_v},\
                 w_d:{alpha},b:{bs},epochs:{epochs},{winit},{optimizer_name},{g_activation}"

    #Preparing hyperparameters for our optimizer
    if optimizer_name == 'sgd':
        optim_hp = {'learning_rate':lr, 'alpha': alpha}
    elif optimizer_name == 'momentum' or optimizer_name == 'nag' or optimizer_name == 'rmsprop':
        optim_hp = {'learning_rate':lr, 'beta': beta, 'alpha': alpha}
    elif optimizer_name == 'adam' or optimizer_name == 'nadam':
        optim_hp = {'learning_rate':lr, 'beta_m': beta_m, 'beta_v': beta_v, 'alpha':alpha}

    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    hidden_sizes = [hidden_size]*num_layers + [n_classes]

    network_hp = {'batch_size': bs, 'g_activation': ACTIVATIONS_MAP[g_activation]}
    mynn = MyNeuralNetwork(n_features, hidden_sizes, n_classes)
   
    mynn.update_network_hp(**network_hp)  
    mynn.update_optim_hp(optimizer_name,**optim_hp)  
    mynn.params_init(way = winit)  
    mynn.train(X_train,y_train,X_valid,y_valid,epochs = epochs)

    mynn.clear_opt_history()
