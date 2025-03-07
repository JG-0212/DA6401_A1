from Data import *
from NeuralNetwork import *
from Optimizers import *
import yaml
import wandb

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels)   =  load_data()
    train_images, train_labels, valid_images, valid_labels = split_validation(train_images, train_labels, valid_size=0.2, seed = 42)
    X_train, X_valid, X_test = preprocess(train_images), preprocess(valid_images), preprocess(test_images)
    y_train, y_valid, y_test = one_hot_encoded(train_labels), one_hot_encoded(valid_labels), one_hot_encoded(test_labels)

    n_features = X_train.shape[1]
    n_classes = 10
    hidden_sizes = [15,15,n_classes]

    mynn = MyNeuralNetwork(n_features, hidden_sizes, n_classes)
    mynn.params_init()

    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    run = wandb.init(entity="jayagowtham-indian-institute-of-technology-madras",project = 'DA6401',config=config)
    # Note that we define values from `wandb.config`
    # instead of  defining hard values
    lr = wandb.config.learning_rate
    beta = wandb.config.momentum
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs
    optimizer_name = wandb.config.optimizer
    run.name = f"lr:{lr},beta:{beta},bs:{bs},epochs:{epochs},opt:{optimizer_name}"
    if optimizer_name == 'gd':
        optim_hp = {'learning_rate':lr}
    elif optimizer_name == 'mgd' or optimizer_name == 'nag':
        optim_hp = {'learning_rate':lr, 'beta': beta}

    mynn.update_network_hp(**{'batch_size':bs})  
    mynn.update_optim_hp(optimizer_name,**optim_hp)    
    mynn.train(X_train[:100],y_train[:100],X_valid[:100],y_valid[:100],epochs = epochs)
