import random

START_SYMBOL = '<S>'
random.seed(1234)

def to_n_gram(word, ngram_range, dropout_probability = 0):
    """
    Turns a word into a list of ngrams.
    :param word: the word
    :param ngram_range: a list with two elements: the minimum and maximum number of characters per ngram
    :return: the corresponding list of ngrams
    """

    assert len(ngram_range) == 2

    n_min = ngram_range[0]
    n_max = ngram_range[1]
    ngrams = []

    letters = [START_SYMBOL] + list(word) + [START_SYMBOL]

    for i in range(len(letters)):
        for j in range(i + n_min, min(len(letters) + 1, i + n_max + 1)):
            ngram = ''.join(letters[i:j])
            ngrams.append(ngram)

    if dropout_probability > 0:
        ngrams = [ngram for ngram in ngrams if random.random() < (1 - dropout_probability)]

    if not ngrams:
        ngrams = ['UNK']
    return ngrams
