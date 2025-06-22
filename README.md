
## Steps

### Initialize parameters

### Run the optimization loop
#### Forward propagation to compute the loss function
#### Backward propagation to compute the gradients with respect to the loss function
#### Clip the gradients to avoid exploding gradients
#### Using the gradients, update your parameters with the gradient descent update rule.

### Return the learned parameters



# RNN_from_scratch (for Chemical Compound Name Generation)
![Alt text describing the image](images/1_QfHHiQuifJ-W9vQblQCj0Q.png)

This project implements a character-level Recurrent Neural Network (RNN) from scratch using only NumPy.
The RNN is trained on a dataset of names and can generate new, plausible names character-by-character.
It includes data preprocessing, forward and backward passes, parameter updates, and sampling from the trained model.
No deep learning libraries are used to help understand the inner workings of RNNs.

---

## Features

- Basic RNN implementation from scratch using NumPy
- Training with Backpropagation Through Time (BPTT)
- Character-level sequence generation
- One-hot encoding of input characters
- Gradient clipping to prevent exploding gradients
- Cross-entropy loss with softmax output layer

---

## Main Components

- `rnn_forward`: Forward pass through the entire input sequence  
- `rnn_backward`: Backpropagation through the entire sequence  
- `rnn_cell_backward`: Backpropagation for one time step  
- `initialize_parameters`: Initialize model weights and biases  
- `optimize`: Performs one training step (forward + backward + parameter update)  
- `model`: Trains the model over multiple iterations and prints generated samples periodically  
- `utils.py`: Utility functions (softmax, gradient clipping, sampling, smoothing, etc.)  
- `preprocessing.py`: Data loading and preparation (character-to-index mapping)

---

## Installation

This project requires Python 3.x and NumPy.

## Usage

* Prepare a text file with a list of names (one name per line).
* Set the data_path variable in the main script to your data file location.
* Run the training script:
```bash
python train_test.py
```
## Configurable Parameters

* num_iterations: Number of training iterations (e.g., 30000)
* n_a: Size of the hidden state layer (e.g., 70 or 100)
* learning_rate: Learning rate for gradient descent (e.g., 0.001)
* dino_names: Number of generated names printed during training intervals

## Sample Output
```yaml
Iteration: 0, Loss: 23.080753

nkknzexbw
kknzexbw
knzexbw
nzexbw
zexbw
exbw
xbw
bw
w  


Iteration: 22000, Loss: 15.823621
onericoin
onigane
osicoflactin
wecogen
ganenide
aneone
oisoit
coreol
oneon
rasic
```

## Technical Details
* Weight matrices Wax, Waa, and Wya represent input-to-hidden, hidden-to-hidden, and hidden-to-output connections, respectively
* Bias vectors ba and by
* Hidden layer uses tanh activation function
* Output layer uses softmax to produce probability distribution over characters
* Cross-entropy loss is computed at each time step
* Training is done via gradient descent with gradient clipping for stability

