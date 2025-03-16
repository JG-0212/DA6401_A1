from Data import *
from NeuralNetwork import *
from Optimizers import *
from Activations import *
import yaml
import wandb
import argparse

if __name__ == '__main__':
    #Loading and processing the data
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp","--wandb_project", default = "DA6401", type=str)
    parser.add_argument("-we","--wandb_entity", default = "jayagowtham-indian-institute-of-technology-madras", type=str)
    parser.add_argument("-d","--dataset", default = "fashion_mnist", type=str)
    parser.add_argument("-e","--epochs", default = 1, type=int)
    parser.add_argument("-b","--batch_size", default = 4, type=int)  
    parser.add_argument("-l","--loss", default = "cross_entropy", type=str)
    parser.add_argument("-o","--optimizer", default = "sgd", type=str)
    parser.add_argument("-lr","--learning_rate", default = 0.1, type=float)
    parser.add_argument("-m","--momentum", default = 0.5, type=float)  
    parser.add_argument("-beta","--beta", default = 0.9, type=float)   
    parser.add_argument("-beta1","--beta1", default = 0.9, type=float)    
    parser.add_argument("-beta2","--beta2", default = 0.99, type=float)   
    parser.add_argument("-eps","--epsilon", default = 1e-6, type=float)   
    parser.add_argument("-w_d","--weight_decay", default = .0, type=float)
    parser.add_argument("-w_i","--weight_init", default = "random", type=str)
    parser.add_argument("-nhl","--num_layers", default = 1, type=int)
    parser.add_argument("-sz","--hidden_size", default = 4, type=int)    
    parser.add_argument("-a","--activation", default = "sigmoid", type=str) 

    args = parser.parse_args()  

    (train_images, train_labels), (test_images, test_labels)   =  load_data(args.dataset)
    train_images, train_labels, valid_images, valid_labels = split_validation(train_images, train_labels, valid_size=0.1, seed = 42)

    # train_images, train_labels,_,_ = split_validation(train_images, train_labels, valid_size=0.95, seed = 42)
    X_train, X_valid, X_test = preprocess(train_images), preprocess(valid_images), preprocess(test_images)
    y_train, y_valid, y_test = one_hot_encoded(train_labels), one_hot_encoded(valid_labels), one_hot_encoded(test_labels)    
    wp = args.wandb_project
    we = args.wandb_entity             
    run = wandb.init(project = wp, entity = we)
    if args.dataset == 'fashion_mnist':
        get_class_sample(train_images, train_labels, class_names=fashion_names, wb_verbose=True)
    elif args.dataset == 'mnist':
        get_class_sample(train_images, train_labels, class_names=number_names, wb_verbose=True)

    #Getting the configutation from wandb sweep
    loss = args.loss
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    lr = args.learning_rate
    m = args.momentum
    beta = args.beta
    beta_m = args.beta1
    beta_v = args.beta2
    alpha = args.weight_decay
    winit = args.weight_init
    bs = args.batch_size
    epochs = args.epochs
    optimizer_name = args.optimizer
    g_activation = args.activation
    epsilon = args.epsilon

    # #Renaming our run
    run.name = "Full_data_best_hp_run"

    #Preparing hyperparameters for our optimizer
    if optimizer_name == 'sgd':
        optim_hp = {'learning_rate':lr, 'alpha': alpha}
    elif optimizer_name == 'momentum' or optimizer_name == 'nag':
        optim_hp = {'learning_rate':lr, 'beta': m, 'alpha': alpha}
    elif optimizer_name == 'rmsprop':
        optim_hp = {'learning_rate':lr, 'beta': beta, 'alpha': alpha, 'epsilon':epsilon}
    elif optimizer_name == 'adam' or optimizer_name == 'nadam':
        optim_hp = {'learning_rate':lr, 'beta_m': beta_m, 'beta_v': beta_v, 'alpha':alpha, 'epsilon':epsilon}

    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    hidden_sizes = [hidden_size]*num_layers + [n_classes]

    network_hp = {'batch_size': bs, 'g_activation': ACTIVATIONS_MAP[g_activation], 'loss': loss}
    mynn = MyNeuralNetwork(n_features, hidden_sizes, n_classes)
    mynn.update_network_hp(**network_hp)  
    mynn.update_optim_hp(optimizer_name,**optim_hp)  
    mynn.params_init(way = winit)  
    mynn.train(X_train,y_train,X_valid,y_valid,epochs = epochs)

    #Test performance
    y_test_hat = mynn.predict(X_test)
    print(f"The accuracy on test set is {np.sum(np.argmax(y_test_hat,axis = 1)==np.argmax(y_test,axis = 1))/y_test.shape[0]}")

    y_test_hat_ohe = np.eye(y_test_hat.shape[1])[np.argmax(y_test_hat,axis = 1)]
    plot_cm(y_test_hat_ohe, y_test, class_names = fashion_names, wb_verbose = True)

    mynn.clear_opt_history()
