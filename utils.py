import pandas as np
import copy

def clip(gradients, maxValue):
    gradients = copy.deepcopy(gradients)
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients


def softmax(x):
    exps = np.exp(x - np.max(x))  # for numerical stability
    return exps / np.sum(exps, axis=0)


# Load text and preprocess
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    chars = sorted(list(set(data)))
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}
    return data, char_to_ix, ix_to_char

def one_hot(char_idx, vocab_size):
    vec = np.zeros((vocab_size, 1))
    vec[char_idx] = 1
    return vec