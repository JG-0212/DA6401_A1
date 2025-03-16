import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import fashion_mnist, mnist
import seaborn as sns
import wandb

fashion_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
number_names = ['0','1','2','3','4','5','6','7','8','9']

def load_data(name = 'fashion_mist'):
    if name == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    elif name == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()       
    return (train_images, train_labels), (test_images, test_labels)


def plot_cm(y_pred, y_true, class_names = fashion_names, wb_verbose = False):
    mat = (y_true.T@y_pred).astype(int)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    if wb_verbose == True:
        wandb.log({"Confusion_matrix_report":wandb.Image(plt.gcf())})
    else:
        plt.show()
    plt.clf

def split_validation(X,y, valid_size = 0.1, seed = 42):
    X_train, y_train, X_test, y_test = [],[],[],[]
    for i,ci in enumerate(np.unique(y)):
        indices = np.where(y==ci)[0]
        train_len = int(indices.shape[0]*(1-valid_size))
        shuffled = np.random.RandomState(seed+i).permutation(indices)
        X_train.append(X[shuffled[:train_len]])
        y_train.append(y[shuffled[:train_len]])
        X_test.append(X[shuffled[train_len:]])
        y_test.append(y[shuffled[train_len:]])
    return np.vstack(X_train), np.hstack(y_train), np.vstack(X_test), np.hstack(y_test)

def get_class_sample(data,labels,class_names = fashion_names, wb_verbose = False):
    classes = np.unique(labels)
    n_classes = classes.shape[0]

    rows,cols = int(np.sqrt(n_classes)), int(np.sqrt(n_classes))
    rows  = rows + 1 if cols**2!=n_classes else rows
    _,axes = plt.subplots(rows,cols, figsize = (6,6))
    for ax in axes.ravel():
        ax.set_axis_off()
    plt.axis('off')
    i,j = 0,0
    for ind,c in enumerate(classes):
        class_name = class_names[ind]
        idx = np.where(labels==c)[0][0]
        image = data[idx]
        axes[i,j].imshow(image,cmap = 'gray')
        axes[i,j].set_title(class_name)
        j += 1
        if(j>=cols):
            i += 1
            j = 0
    if wb_verbose == True:
        wandb.log({"Sample_report":wandb.Image(plt.gcf())})
    else:
        plt.show()
    plt.clf()

def preprocess(images):
    images = images.reshape((images.shape[0],-1))
    images = images.astype('float32')/255.
    return images

def one_hot_encoded(labels):
    n_classes = np.size(np.unique(labels))
    out = np.zeros((labels.shape[0], n_classes))
    for i,l in enumerate(labels):
        out[i,l] = 1
    return out