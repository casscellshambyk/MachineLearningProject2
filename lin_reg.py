import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

data_dir = 'kc_house_data.csv'

# Importing DataSet
dataset = pd.read_csv(data_dir)

# Splitting the data into Train and Test
train = dataset.sample(frac=0.67)
test = dataset.drop(train.index)

# translate datasets into numpy arrays and separate pixel data from labels
X_train = train['sqft_living'].values.astype('int64').reshape(-1, 1)
ones = np.ones([X_train.shape[0], 1])  # create a array containing only ones
X_train = np.concatenate([ones, X_train], 1)  # cocatenate the ones to X matrix

y_train = train['price'].values.astype('int64').reshape(-1, 1)  # labels

X_test = test['sqft_living'].values.astype('int64').reshape(-1, 1)
y_test = test['price'].values.astype('int64').reshape(-1, 1)  # labels

alpha = .0000001  # maximum alpha I could get to work
n_iters = 100  # Tests ran with n_iters = 500000000 on google collabs to allow convergence from rand starting point

theta = np.random.randn(2, 1)


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    cost_history = cal_cost(theta, X, y) + 300
    curr_cost = cal_cost(theta, X, y)
    # theta_history = np.zeros((iterations, 2))
    conv = False
    it = 0
    while not conv:
        prediction = np.dot(X, theta)

        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        # theta_history[it, :] = theta.T
        curr_cost = cal_cost(theta, X, y)
        if it % 500 == 0:
            print(curr_cost)
        if curr_cost >= cost_history:
            conv = True
        else:
            cost_history = curr_cost
            iterations -= 1
            it += 1
            if iterations < 0:
                break

    return theta, curr_cost


def cal_cost(thet, X, y):
    m = len(y)

    predictions = X.dot(thet)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


theta, cost = gradient_descent(X_train, y_train, theta, alpha, n_iters)

print(theta)
print(cost)
