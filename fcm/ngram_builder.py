import random
import itertools
import numpy as np
from collections import Counter
from typing import List

import my_log

logger = my_log.get_logger('root')

START_SYMBOL = '<S>'
UNK_TOKEN = 'UNK'
PAD_TOKEN = 'PAD'

UNK_ID = 0
PAD_ID = 1


class NGramFeatures:
    def __init__(self, ngrams: List[str], ngram_ids: List[int]):
        self.ngrams = ngrams
        self.ngram_ids = ngram_ids
        self.ngrams_length = len(ngram_ids)

    def __repr__(self):
        return '{}, {}, {}'.format(self.ngrams, self.ngram_ids, self.ngrams_length)


class BatchedNGramFeatures:
    def __init__(self, ngram_ids: np.ndarray, ngrams_length: np.ndarray):
        self.ngram_ids = ngram_ids
        self.ngrams_length = ngrams_length

    def __repr__(self):
        return '{}, {}'.format(self.ngram_ids, self.ngrams_length)


class NGramBuilder:
    def __init__(self, vocab_file: str, ngram_threshold: int = 4, nmin: int = 3, nmax: int = 5):

        self.nmin = nmin
        self.nmax = nmax

        self.ngram2id = {UNK_TOKEN: UNK_ID, PAD_TOKEN: PAD_ID}
        self.id2ngram = [UNK_TOKEN, PAD_TOKEN]

        ngram_counts = Counter()

        with open(vocab_file, 'r', encoding='utf8') as file:

            for line in file:
                word = line.split()[0]
                ngram_counts.update(self.to_n_gram(word, self.nmin, self.nmax))

        for (ngram, count) in ngram_counts.most_common():
            if count >= ngram_threshold:
                id = len(self.id2ngram)
                self.ngram2id[ngram] = id
                self.id2ngram.append(ngram)

        logger.info('Found a total of {} ngrams with a minimum count of {} and (nmin,nmax)=({},{})'.format(
            len(self.id2ngram), ngram_threshold, nmin, nmax))

    def get_ngram_features(self, word: str, dropout_probability: float = 0) -> NGramFeatures:

        ngrams = self.to_n_gram(word, self.nmin, self.nmax, dropout_probability)
        ngram_ids = [self.ngram2id[ngram] if ngram in self.ngram2id else UNK_ID for ngram in ngrams]
        return NGramFeatures(ngrams, ngram_ids)

    def batchify(self, features: List[NGramFeatures]) -> BatchedNGramFeatures:
        ngram_ids = np.array(
            list(itertools.zip_longest(*[x.ngram_ids for x in features], fillvalue=PAD_ID)),
            dtype=np.int32).T
        ngrams_length = np.array([x.ngrams_length for x in features], dtype=np.int32)
        return BatchedNGramFeatures(ngram_ids, ngrams_length)

    @staticmethod
    def to_n_gram(word: str, nmin: int, nmax: int, dropout_probability: float = 0) -> List[str]:
        """
        Turns a word into a list of n-grams.
        :param word: the word
        :param nmin: the minimum number of characters per n-gram
        :param nmax: the maximum number of characters per n-gram
        :param dropout_probability: the probability of randomly removing an n-gram
        :return: the list of n-grams
        """
        ngrams = []

        letters = [START_SYMBOL] + list(word) + [START_SYMBOL]

        for i in range(len(letters)):
            for j in range(i + nmin, min(len(letters) + 1, i + nmax + 1)):
                ngram = ''.join(letters[i:j])
                ngrams.append(ngram)

        if dropout_probability > 0:
            ngrams = [ngram for ngram in ngrams if random.random() < (1 - dropout_probability)]

        if not ngrams:
            ngrams = [UNK_TOKEN]
        return ngrams

    def get_number_of_ngrams(self) -> int:
        return len(self.id2ngram)
