import collections
import numpy as np
import re

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample)-n+1):
                yield sample[i:i+n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5*(kl_p_m + kl_q_m) / np.log(2)

def load_dataset(path, max_length, tokenize=False, max_vocab_size=2048):
    """
    Load and process the dataset from a file.
    
    Args:
        path: Path to the dataset file
        max_length: Maximum length of each line
        tokenize: Whether to tokenize the lines
        max_vocab_size: Maximum vocabulary size
    
    Returns:
        filtered_lines: List of processed lines
        charmap: Dictionary mapping characters to indices
        inv_charmap: List mapping indices to characters
    """
    lines = []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove newline character
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                # Truncate if longer than max_length
                if len(line) > max_length:
                    line = line[:max_length]
                
                # Pad shorter sequences with ` character
                if len(line) < max_length:
                    line = line + (('`',) * (max_length-len(line)))
                
                lines.append(line)

        if not lines:
            raise ValueError(f"No valid lines found in file: {path}")

        # Shuffle the lines
        np.random.shuffle(lines)

        # Count character frequencies
        counts = collections.Counter(char for line in lines for char in line)

        # Create character mappings
        charmap = {'unk': 0}
        inv_charmap = ['unk']

        # Add characters by frequency
        for char, count in counts.most_common(max_vocab_size-1):
            if char not in charmap:
                charmap[char] = len(inv_charmap)
                inv_charmap.append(char)

        # Filter lines using the character map
        filtered_lines = []
        for line in lines:
            filtered_line = []
            for char in line:
                filtered_line.append(char if char in charmap else 'unk')
            filtered_lines.append(tuple(filtered_line))

        print(f"Loaded {len(filtered_lines)} lines in dataset")
        print(f"Vocabulary size: {len(charmap)}")
        
        return filtered_lines, charmap, inv_charmap

    except FileNotFoundError:
        print(f"Error: Could not find file at path: {path}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise