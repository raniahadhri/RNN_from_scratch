from RNN_models import rnn_forward, rnn_backward, update_parameters
from utils import clip, sample, get_sample, smooth
from preprocessing import preprocessing
import numpy as np
import pickle



def initialize_parameters(n_a, n_x, n_y):
    np.random.seed(1)
    parameters = {}
    parameters['Wax'] = np.random.randn(n_a, n_x) * 0.01  # (100, 27)
    parameters['Waa'] = np.random.randn(n_a, n_a) * 0.01  # (100, 100)
    parameters['Wya'] = np.random.randn(n_y, n_a) * 0.01  # (27, 100)
    parameters['ba'] = np.zeros((n_a, 1))                  # (100, 1)
    parameters['by'] = np.zeros((n_y, 1))                 # (27, 1)
    return parameters


def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, learning_rate)
    return loss, gradients, a[len(X)-1]




def model(data_x, ix_to_char, char_to_ix, num_iterations = 30000, n_a = 60, dino_names = 10, vocab_size = 27, verbose = False):
    n_x, n_y = vocab_size, vocab_size
    
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = -np.log(1.0 / vocab_size) * 7  # 7 est une longueur moyenne de nom

    examples = [x.strip() for x in data_x]
    np.random.seed(0)
    np.random.shuffle(examples)
    a_prev = np.zeros((n_a, 1))
    last_name = "abc"

    for j in range(num_iterations):
        
        idx = j % len(examples)
        single_example = examples[idx]
        single_example_chars = [ch for ch in single_example]
        single_example_ix = [char_to_ix[ch] for ch in single_example_chars]
        X = [None] + single_example_ix
        
        ix_newline = char_to_ix['\n']
        Y = single_example_ix + [ix_newline]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.001 )

        if verbose and j % 2000 == 0:
            print("j =", j, "idx =", idx)
        if verbose and j in [0]:
            print("single_example =", single_example)
            print("single_example_chars", single_example_chars)
            print("single_example_ix", single_example_ix)
            print(" X = ", X, "\n", "Y =       ", Y, "\n")
        
        loss = smooth(loss, curr_loss)
        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            seed = 0
            for name in range(dino_names):
                seed = name + j  
                sampled_indices = sample(parameters, char_to_ix, seed)
                last_name = get_sample(sampled_indices, ix_to_char)
                print(last_name.replace('\n', ''))
                seed += 1 
      
            print('\n')
        
    return parameters, last_name


if __name__ == "__main__":
    #data_path = "data/dino.txt"
    data_path = "data/natural_compound_names.txt"
    data, char_to_ix, ix_to_char = preprocessing(data_path)
    
    parameters, last_name = model(
        data.split("\n"), ix_to_char, char_to_ix, num_iterations=22001, verbose=True
    )
    
    # Save model
    with open("rnn_model.pkl", "wb") as f:
        pickle.dump(parameters, f)
    
    print("Model saved as rnn_model.pkl")