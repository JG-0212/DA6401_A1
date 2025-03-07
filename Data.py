import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

def load_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return (train_images, train_labels), (test_images, test_labels) 

def split_validation(X,y, valid_size = 0.2, seed = 42):
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

def get_class_sample(data,labels,class_names):
    classes = np.unique(labels)
    n_classes = classes.shape[0]

    rows,cols = int(np.sqrt(n_classes)), int(np.sqrt(n_classes))
    rows  = rows + 1 if cols**2!=n_classes else rows
    fig,axes = plt.subplots(rows,cols, figsize = (6,6))
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
    plt.show()

def preprocess(images):
    images = images.reshape((images.shape[0],-1))
    images = images/255
    return images

def one_hot_encoded(labels):
    n_classes = np.size(np.unique(labels))
    out = np.zeros((labels.shape[0], n_classes))
    for i,l in enumerate(labels):
        out[i,l] = 1
    return out