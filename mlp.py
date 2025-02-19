import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

# may have trouble with good performance from MNIST; anywhere >60% reasonable
# challenges, how did you address
# report should be short
np.random.seed(None)

# tqdm
# just don't use python 3.13
# tested & run with python 3.12.1
def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    rng = np.random.default_rng()
    indices = rng.permutation(train_x.shape[0])
    shuffled_x = train_x[indices]
    shuffled_y = train_y[indices]
    
    # Determine the number of batches needed.
    num_batches = int(np.ceil(train_x.shape[0] / batch_size))
    
    # Split the shuffled arrays into batches without an explicit loop.
    batch_x = np.array_split(shuffled_x, num_batches)
    batch_y = np.array_split(shuffled_y, num_batches)
    
    return (batch_x, batch_y)


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        return x * (1 - x)
    
    
class Tanh(ActivationFunction):
    def forward(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2


class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(x, 0)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    def forward(self, x):
        x_exp = np.exp(x)
        partition = x_exp.sum(1, keepdims=True)
        return x_exp / partition
    
    def derivative(self, x, delta):
        batch_size, num_classes = x.shape
        jacobian = [num_classes, batch_size]
        for i in range(batch_size):
            s_i = x[i].reshape[-1, 1]
            jacobian[i] = np.diagflat(s_i) - (s_i @ s_i.T)
        return jacobian


class Linear(ActivationFunction):
    def forward(self, x):
        return x
    
    def derivative(self, x):
        # return 1
        return np.ones_like(x)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction): # regression problem
    def loss(self, y_true, y_pred): # calculate mean when you do gradient update
        return 0.5 * ((y_pred - y_true) ** 2)
    
    def derivative(self, y_true, y_pred):
        return y_pred - y_true


class CrossEntropy(LossFunction): # classification problem
    def loss(self, y_true, y_pred):
        return -np.log(y_pred[range(len(y_pred)), y_true])
    
    def derivative(self, y_true, y_pred):
        return (-y_true / y_pred) - ((1 - y_true) / (1 - y_pred))


class Layer: # iterate over layers, vectorize neurons
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None
        # stores inputs to this layer
        self.inputs = None

        # Initialize weights and biaes
        # initialize weights with glorot uniform
        self.W = self.glorot_uniform()  # weights
        self.b = np.zeros(fan_out) # biases
        
    def glorot_uniform(self):
        shape = self.fan_in, self.fan_out
        sd = np.sqrt(6.0 / (self.fan_in + self.fan_out))
        rng = np.random.default_rng()
        return rng.uniform(-sd, sd, size=shape)

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        # compute dot product of h & weights + apply biases
        # dot product note: same num of rows as 1st m, same num of cols as 2nd m
        # shape: rows x cols
        self.inputs = h
        pre_activations = h@self.W + self.b
        
        self.activations = self.activation_function.forward(pre_activations)
        

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        # operator * does element-wise multiplication for ndarrays aka hadamard product
        d_activation_function = self.activation_function.derivative(self.activations)
        if (isinstance(self.activation_function, Softmax)):
            dL_dz = delta @ d_activation_function
            # dL_dz =  np.einsum('bij, bj -> bi'), d_activation_function, delta)
        else:
            dL_dz = delta * d_activation_function
            
        self.delta = dL_dz@self.W.T
        
        dL_dW = self.inputs.T@dL_dz
        dL_db = np.sum(dL_dz, axis=0)
        
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []
        
        x = input_data
        delta = loss_grad
        for layer in reversed(self.layers):
            dl_dw, dl_db = layer.backward(x, delta)
            dl_dw_all.append(dl_dw)
            dl_db_all.append(dl_db)
            delta = layer.delta

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = []
        validation_losses = []
        
        # epoch = # of times to run
        # batches = splits of input data sets
       
        # run val only after train
        
        for epoch in range(epochs):
            train_batches_x, train_batches_y = batch_generator(train_x, train_y, batch_size)
            avg_loss = 0.0
            for i in range(len(train_batches_x)):
                out = self.forward(train_batches_x[i])
                
                loss_grad = loss_func.derivative(train_batches_y[i], out)

                train_dl_dw, train_dl_db = self.backward(loss_grad, out)
                
                for layer, dl_dw, dl_db in zip(self.layers, reversed(train_dl_dw), reversed(train_dl_db)):
                    layer.W -= learning_rate * dl_dw
                    layer.b -= learning_rate * dl_db
                
                loss = loss_func.loss(train_batches_y[i], out).sum()
                avg_loss+=loss
                
                
            avg_loss/=len(train_batches_x)
            training_losses.append(avg_loss)

            val_out = self.forward(val_x)
            val_loss = loss_func.loss(val_y, val_out).sum()
            validation_losses.append(val_loss)
            
            print("epoch: ", epoch, "training_loss: ", avg_loss, "validation_loss: ", val_loss)
        
        return training_losses, validation_losses