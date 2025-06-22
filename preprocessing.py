import numpy as np

def preprocessing(data_path):
    data = open(data_path, 'r').read()
    data= data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    chars = sorted(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    return data, char_to_ix, ix_to_char



if __name__ == "__main__":
    file_path="data/natural_compound_names.txt"
    data, char_to_ix, ix_to_char = preprocessing(file_path)
    print(f"names: {data, char_to_ix, ix_to_char}")

