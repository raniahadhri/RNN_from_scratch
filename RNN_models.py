import numpy as np
from utils import softmax


def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, p) = cache
    dtanh = (1 - a_next ** 2) * da_next
    dWax = np.dot(dtanh, xt.T)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims=True)
    dxt = np.dot(p["Wax"].T, dtanh)
    da_prev = np.dot(p["Waa"].T, dtanh)

    gradients = {
        "dxt": dxt, "da_prev": da_prev,
        "dWax": dWax, "dWaa": dWaa, "dba": dba
    }
    return gradients 


def rnn_forward(X, Y, a_prev, parameters, vocab_size=27):
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a_prev)
    loss = 0
    caches = []

    for t in range(len(X)):
        # Step 1: Create one-hot vector for x[t]
        x_t = np.zeros((vocab_size, 1))
        x_t[X[t]] = 1
        x[t] = x_t

        # Step 2: Forward step
        a[t] = np.tanh(np.dot(parameters["Wax"], x[t]) + 
                       np.dot(parameters["Waa"], a[t-1]) + 
                       parameters["ba"])

        z = np.dot(parameters["Wya"], a[t]) + parameters["by"]
        y_hat[t] = softmax(z)

        # Step 3: Compute cross-entropy loss
        loss -= np.log(y_hat[t][Y[t], 0])

        # Store cache for backprop
        caches.append((a[t], a[t-1], x[t], parameters))

    cache = (caches, x, a, y_hat, Y)
    return loss, cache


def rnn_backward(X, Y, parameters, cache):

    (caches_list, x, a, y_hat, Y) = cache
    n_a, m = a[0].shape
    n_x = x[0].shape[0]
    T_x = len(X)

    # Initialize gradients
    dWax = np.zeros_like(parameters["Wax"])
    dWaa = np.zeros_like(parameters["Waa"])
    dba = np.zeros_like(parameters["ba"])
    dWya = np.zeros_like(parameters["Wya"])
    dby = np.zeros_like(parameters["by"])
    da_prevt = np.zeros((n_a, 1))

    # Initialize dx if needed
    dx = {}

    # Loop over time steps backward
    for t in reversed(range(T_x)):
        # Gradient of loss w.r.t. y_hat (softmax + cross-entropy)
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1  # derivative of softmax loss

        # Gradients for output layer
        dWya += np.dot(dy, a[t].T)
        dby += dy

        # Backprop into hidden state
        da = np.dot(parameters["Wya"].T, dy) + da_prevt

        # Call rnn_cell_backward
        gradients = rnn_cell_backward(da, caches_list[t])

        dx[t] = gradients["dxt"]
        dWax += gradients["dWax"]
        dWaa += gradients["dWaa"]
        dba += gradients["dba"]
        da_prevt = gradients["da_prev"]

    # Pack results
    gradients = {
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba,
        "dWya": dWya,
        "dby": dby,
        "da0": da_prevt,
        "dx": dx
    }

    return gradients, a


def update_parameters(parameters, gradients, learning_rate):

    parameters["Wax"] -= learning_rate * gradients["dWax"]
    parameters["Waa"] -= learning_rate * gradients["dWaa"]
    parameters["Wya"] -= learning_rate * gradients["dWya"]
    parameters["ba"]  -= learning_rate * gradients["dba"]
    parameters["by"]  -= learning_rate * gradients["dby"]

    return parameters


