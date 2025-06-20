import pandas as pd
import numpy as np

import copy
from utils import softmax


def rnn_cell_forward(xt, a_prev, p):
    a_next = np.tanh(np.dot(p["Waa"], a_prev) + np.dot(p["Wax"], xt) + p["ba"])
    yt_pred = softmax(np.dot(p["Wya"], a_next) + p["by"])
    cache = (a_next, a_prev, xt, p)
    return a_next, yt_pred, cache


def rnn_forward(x, a0, p):  #parameters
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = p["Wya"].shape

    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, p)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    caches = (caches, x)

    return a, y_pred, caches


def rnn_cell_backward(da_next, cache, dy):
    (a_next, a_prev, xt, p) = cache
    dtanh = (1 - a_next ** 2) * da_next

    dWax = np.dot(dtanh, xt.T)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = dtanh

    dxt = np.dot(p["Wax"].T, dtanh)
    da_prev = np.dot(p["Waa"].T, dtanh)

    dWya = np.dot(dy, a_next.T)
    dby = dy

    gradients = {
        "dxt": dxt, "da_prev": da_prev,
        "dWax": dWax, "dWaa": dWaa, "dba": dba,
        "dWya": dWya, "dby": dby
    }
    return gradients


def rnn_backward(y_pred, y_true, caches):
    (caches_list, x) = caches
    n_a, m, T_x = caches_list[0][0].shape[0], y_true.shape[1], y_true.shape[2]

    grads = {
        "dWax": np.zeros_like(caches_list[0][3]["Wax"]),
        "dWaa": np.zeros_like(caches_list[0][3]["Waa"]),
        "dba": np.zeros_like(caches_list[0][3]["ba"]),
        "dWya": np.zeros_like(caches_list[0][3]["Wya"]),
        "dby": np.zeros_like(caches_list[0][3]["by"]),
    }

    da_next = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        dy = y_pred[:, :, t] - y_true[:, :, t]
        grad = rnn_cell_backward(da_next, caches_list[t], dy)
        for k in grads:
            grads[k] += grad[k]
        da_next = grad["da_prev"]

    return grads


def update_parameters(p, grads, lr):
    for key in grads:
        p[key[1:]] -= lr * grads[key]
    return p

def compute_loss(y_pred, y_true):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

def sample(p, char_to_ix, ix_to_char, seed):
    vocab_size = len(char_to_ix)
    n_a = p["Waa"].shape[1]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1
    counter = 0
    newline_char = char_to_ix['\n']

    while idx != newline_char and counter < 10:
        a_next = np.tanh(np.dot(p["Waa"], a_prev) + np.dot(p["Wax"], x) + p["ba"])
        z = np.dot(p["Wya"], a_next) + p["by"]
        y = softmax(z)

        # Empêcher le modèle de choisir '^' après le premier caractère
        if counter > 0:
            y[char_to_ix['^']] = 0
            y = y / np.sum(y)  # re-normaliser

        np.random.seed(counter + seed)
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a_next

        counter += 1


    return indices