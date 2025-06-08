import numpy as np

def build_vocab(data):
    data = "\n".join(data) + '\n'
    data = data.lower()  # lowercase all characters first
    chars = sorted(list(set(data)))
    char_to_ix = { ch:i for i, ch in enumerate(chars) }
    ix_to_char = { i:ch for i, ch in enumerate(chars) }
    return chars, char_to_ix, ix_to_char

def names_to_sequences(names, char_to_ix):
    vocab_size = len(char_to_ix)
    sequences = []
    for name in names:
        name = name.strip().lower() + '\n'  # lowercase here
        seq_length = len(name) - 1
        X = np.zeros((vocab_size, 1, seq_length))
        Y = np.zeros((vocab_size, 1, seq_length))
        for t in range(seq_length):
            X[char_to_ix[name[t]], 0, t] = 1
            Y[char_to_ix[name[t+1]], 0, t] = 1
        sequences.append((X, Y))
    return sequences

def load_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

def preprocessing(filename):
    names = load_data_from_file(filename)
    chars, char_to_ix, ix_to_char = build_vocab(names)
    sequences = names_to_sequences(names, char_to_ix)
    return chars, sequences, names

if __name__ == "__main__":
    names = load_data_from_file("data/data.txt")
    chars, char_to_ix, ix_to_char = build_vocab(names)
    sequences = names_to_sequences(names, char_to_ix)

    print(f"Vocab: {chars}")
    print(f"First name example: '{names[0]}'")
    print(f"Input shape: {sequences[0][0].shape}")
    print(f"Output shape: {sequences[0][1].shape}")