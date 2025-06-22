import pandas as pd
import numpy as np
import copy


def clip(gradients, maxValue):
    for key in gradients:
        if isinstance(gradients[key], dict):
            # Recursively clip inside nested dicts
            gradients[key] = clip(gradients[key], maxValue)
        else:
            gradients[key] = np.array(gradients[key])
            np.clip(gradients[key], -maxValue, maxValue, out=gradients[key])
    return gradients


def softmax(z, temperature=1.0):
    z = z / temperature
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


#--------------------------------------------------------------------------------------------------------------

def sample(parameters, char_to_ix, seed, temperature=1.0):
    vocab_size = len(char_to_ix)
    n_a = parameters["Waa"].shape[1]

    x = np.zeros((vocab_size, 1))
    x[char_to_ix['\n']] = 1

    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']

    while idx != newline_character and counter < 50:
        a = np.tanh(np.dot(parameters["Wax"], x) + np.dot(parameters["Waa"], a_prev) + parameters["ba"])
        z = np.dot(parameters["Wya"], a) + parameters["by"]
        y = softmax(z, temperature=temperature)

        np.random.seed(counter + seed)
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        counter += 1

    if counter == 50:
        indices.append(newline_character)

    return indices





def get_sample(sampled_indices, ix_to_char):
    name = ''.join([ix_to_char[ix] for ix in sampled_indices])
    return name


def smooth(loss, curr_loss, beta=0.999):
    return beta * loss + (1 - beta) * curr_loss
