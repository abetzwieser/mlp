import math
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from mlp import *
import random
import matplotlib.pyplot as plt

# downloads and splits the dataset, instantiates your MLP
# design, and trains the model. The training and validation loss should be printed out at each
# epoch.

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


X_train = X_train.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1) # reshape to be a 2d array
X_val = X_val.to_numpy()
y_val = y_val.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()
y_test = y_test.to_numpy().reshape(-1, 1)

def graph(graph_epochs, training_losses, validation_losses):
    plt.plot(graph_epochs, training_losses, 'g', label='Training loss')
    plt.plot(graph_epochs, validation_losses, 'b', label='Validation loss')

    plt.title('Training and Validation Loss')

    plt.xlabel('Epochs')
    plt.xticks(graph_epochs)

    plt.ylabel('Loss')

    plt.legend()
    plt.show()

# creating mlp
lr = 0.001 # learning rate
batch_size = 20
epochs = 80

act_fn = Relu()
loss_fn = SquaredError()

mlp_layers = (Layer(7, 2, Sigmoid()),
              Layer(2, 1, Tanh()),
              Layer(1, 1, Tanh()))

mlp = MultilayerPerceptron(mlp_layers)
training_losses, validation_losses = mlp.train(X_train, y_train, X_val, y_val, loss_fn, lr, batch_size, epochs)

test_pred = mlp.forward(X_test)
err = (test_pred - y_test)**2
err = err.sum() / (len(test_pred))
print("error: ", err)

print("10 random samples~")
for i in range(10):
    rando = random.randint(0, len(y_test))
    print("predicted: ", test_pred.tolist()[rando], "\t\ttrue: ", y_test[rando])

graph_epochs = range(1, epochs + 1)
graph(graph_epochs, training_losses, validation_losses)


    