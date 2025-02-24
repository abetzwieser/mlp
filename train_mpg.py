import math
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from mlp import *
import random
import matplotlib.pyplot as plt
import copy

def get_mlp_avg_performance(mlp_layers, runs, data, test_data, show_graph):
    """Gives average MLP performance with a given design over x number of runs

    :param mlp_layers: tuple with MLP layers
    :param runs: number of times to run
    :param data: tuple of MLP parameters
    :param test_data: tuple of (test features, test targets)
    :param graph: boolean value indicating whether to display graph for each run
    """
    X_train, y_train, X_val, y_val, loss_fn, lr, batch_size, epochs = data # retrieve MLP params
    X_test, y_test = test_data # retrieve test data
    test_losses = []
    test_err = []
    
    for i in range(runs):
        new_mlp_layers = copy.deepcopy(mlp_layers) # get deep copy of layers (avoid using trained layers)
        mlp = MultilayerPerceptron(new_mlp_layers) # new MLP
        
        # train & predict on test features
        t_losses, v_losses = mlp.train(X_train, y_train, X_val, y_val, loss_fn, lr, batch_size, epochs)
        test_pred = mlp.forward(X_test)
        
        loss = (test_pred - y_test).sum() # compute loss
        err = (test_pred - y_test)**2   # squared error
        err = err.sum() / (len(test_pred))  # getting mean
        
        # get loss & error for this run
        test_losses.append(loss)
        test_err.append(err)
        
        if show_graph: # graph this run's training/validation curves
            graph_epochs = range(1, epochs + 1)
            graph(graph_epochs, t_losses, v_losses)
            
    # calculate & display average error / loss on test set
    test_losses = np.array(test_losses)
    test_err = np.array(test_err)
    avg_loss = test_losses.sum()/len(test_losses)
    avg_err = test_err.sum()/len(test_err)
    print ("average error: ", avg_err,  "average loss: ", avg_loss)

# data preparation borrowed from:
# https://github.com/jghawaly/CSC7809_FoundationModels/blob/main/example_notebooks/linear_regression.ipynb

# fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# data (as pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into one DataFrame for easy filtering
data = pd.concat([X, y], axis=1)

# Drop rows where the target variable is NaN
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

# SPLITTING DATA

# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)    # Standard deviation of each feature

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (targets)
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()    # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

print(f"Samples in Training:   {len(X_train)}")
print(f"Samples in Validation: {len(X_val)}")
print(f"Samples in Testing:    {len(X_test)}")

# Converting pandas dataframes -> 2darrays
X_train = X_train.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1) # reshape to be a 2d array
X_val = X_val.to_numpy()
y_val = y_val.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()
y_test = y_test.to_numpy().reshape(-1, 1)

# creating MLP
lr = 0.001 # learning rate
batch_size = 20
epochs = 90

loss_fn = SquaredError()

mlp_layers = (Layer(7, 2, Sigmoid()),
              Layer(2, 1, Tanh()),
              Layer(1, 1, Tanh()))

layer_dupe = copy.deepcopy(mlp_layers) # get deep copy of layers before training

mlp = MultilayerPerceptron(mlp_layers)
training_losses, validation_losses = mlp.train(X_train, y_train, X_val, y_val, loss_fn, lr, batch_size, epochs)

# testing performance with test set
test_pred = mlp.forward(X_test)
loss = (abs(test_pred - y_test)).sum() # compute loss
err = (test_pred - y_test)**2   # squared error
err = err.sum() / (len(test_pred))  # getting mean
print("error: ", err, "loss: ", loss) # print MSE, loss

# reverse transform values for human readability
test_pred = (test_pred * y_std) + y_mean
test_true = (y_test * y_std) + y_mean

# print 10 random sample test values, comparing MLP prediction v. true value
print("\nHere are 10 random samples!")
for i in range(10):
    rando = random.randint(0, len(y_test)-1)
    print("predicted: ", test_pred.tolist()[rando], "\t\ttrue: ", test_true[rando])

# plot training/validation losses
graph_epochs = range(1, epochs + 1)
graph(graph_epochs, training_losses, validation_losses)

# get average performance of current design over X number of runs
runs = 100
show_graph = False

data = (X_train, y_train, X_val, y_val, loss_fn, lr, batch_size, epochs)
test_data = (X_test, y_test)

#get_mlp_avg_performance(layer_dupe, runs, data, test_data, show_graph)

    