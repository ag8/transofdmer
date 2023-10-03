import pickle

import numpy as np
import tokenmonster

from tokenizer import PreambleTokenizer

vocab_size = 32000
vocab = tokenmonster.load(f"english-{vocab_size}-consistent-v1")

def create_sample_target_pairs(filename, tokenizer, sample_length=100, stride=50):
    """
    Create sample-target pairs using a sliding window.

    Parameters:
    - filename: path to the text file.
    - tokenizer: tokenizer function.
    - sample_length: length of each context+target sample in terms of tokens.
    - stride: step size for the sliding window in terms of tokens.

    Returns:
    - List of (sample, target) tokenized pairs.
    """
    with open(filename, 'r') as file:
        text = file.read()

    tokens = tokenizer(text)

    pairs = []
    for i in range(0, len(tokens) - sample_length, stride):
        sample = tokens[i:i + sample_length]
        target = tokens[i + 1:i + sample_length + 1]  # Offset by one token for prediction
        pairs.append([sample, target])

    return pairs


def save_pairs_to_pickle(pairs, pickle_filename):
    """
    Save tokenized sample-target pairs to a pickle file.

    Parameters:
    - pairs: list of tokenized (sample, target) pairs.
    - pickle_filename: filename to save the pickle.
    """
    with open(pickle_filename, 'wb') as file:
        pickle.dump(pairs, file)


if __name__ == '__main__':
    # Dummy tokenizer function for demonstration, replace with your own
    def dummy_tokenizer(text):
        return PreambleTokenizer.encode_text(text)


    pairs = create_sample_target_pairs('get_low.txt', dummy_tokenizer, sample_length=20,
                                       stride=10)
    save_pairs_to_pickle(np.asarray(pairs), 'tokenized_samples_targets.pkl')
