import numpy as np # linear algebra
import struct
from array import array
from os.path import join
from mlp import *
from sklearn.model_selection import train_test_split
import random

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        image_vectors = []
        for i in range(size):
            images.append([0] * rows * cols)
            image_vectors.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i] = img
            image_vectors[i] = img.flatten(order='C').reshape(-1)       # flatten images into 2d vectors of len 784
        return image_vectors, labels, images
            
    def load_data(self):
        x_train, y_train, train_images = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test, test_images = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train, train_images),(x_test, y_test, test_images)        


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()

#
# Set file paths based on added MNIST Datasets
#
input_path = '.\\mnist'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte\\train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte\\t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')
#
# Load MNIST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train, train_images), (x_test, y_test, test_images) = mnist_dataloader.load_data()

test_labels = y_test

# splitting train into train & val (80 / 20 split)
x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train,
                test_size=0.2,
                random_state=42,    # for reproducibility
                shuffle=True,       # whether to shuffle the data before splitting
        )

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Compute statistics for X (features)
x_mean = x_train.mean(axis=0)  # Mean of each feature
x_std = x_train.std(axis=0)    # Standard deviation of each feature
x_std = np.where(x_std == 0, 1e-8, x_std) # Replace 0s to avoid divide by 0 errors

# Standardize X
x_train = (x_train - x_mean) / x_std
x_val = (x_val - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# One hot encoding
num_classes = 10
y_train = np.eye(num_classes)[y_train]  # Convert labels to one-hot
y_val = np.eye(num_classes)[y_val]
y_test = np.eye(num_classes)[y_test]

# Creating mlp
loss_fn = CrossEntropy()

mlp_layers = (Layer(784, 48, Sigmoid()),
              Layer(48, 24, Tanh()),
              Layer(24, 10, Softmax()))

lr = 0.001 # learning rate
batch_size = 20
epochs = 10

mlp = MultilayerPerceptron(mlp_layers)
training_losses, validation_losses = mlp.train(x_train, y_train, x_val, y_val, loss_fn, lr, batch_size, epochs)

test_pred = mlp.forward(x_test)

pred = test_pred.argmax(axis=1)
y_test = y_test.argmax(axis=1)
correct = (pred == y_test)

accuracy = 100 * correct.sum().astype(np.float64) / len(y_test)
print("accuracy: ", accuracy)

graph_epochs = range(1, epochs + 1)
graph(graph_epochs, training_losses, validation_losses)

images_2_show = []
titles_2_show = []

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(test_images[r])
    titles_2_show.append('prediction: ' + str(pred[r]) + ' true: ' + str(test_labels[r]))
    
show_images(images_2_show, titles_2_show) 