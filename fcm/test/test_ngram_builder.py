import os
import unittest
import numpy as np
import numpy.testing as npt

import my_log
from ngram_builder import NGramBuilder, UNK_TOKEN

logger = my_log.get_logger('root')


class TestNGramBuilder(unittest.TestCase):
    def test_ngram_builder(self):
        vocab_file = os.path.join(os.path.dirname(__file__), 'data', 'vocab.txt')

        ngram_builder = NGramBuilder(vocab_file, ngram_threshold=4, nmin=3, nmax=3)
        self.assertEqual(len(ngram_builder.id2ngram), 6)
        self.assertEqual(set(ngram_builder.get_ngram_features('ngrax').ngrams), {'<S>ng', "ngr", "gra", "rax", "ax<S>"})
        self.assertEqual(set([ngram_builder.id2ngram[x] for x in ngram_builder.get_ngram_features('ngrax').ngram_ids]),
                         {'<S>ng', "ngr", "gra", UNK_TOKEN})
        self.assertEqual(len(ngram_builder.get_ngram_features('ngrax').ngram_ids), 5)

        ngram_builder = NGramBuilder(vocab_file, ngram_threshold=3, nmin=3, nmax=3)
        self.assertEquals(len(ngram_builder.id2ngram), 8)
        self.assertTrue("rd<S>" in ngram_builder.id2ngram)

        features_a = ngram_builder.get_ngram_features('sword')
        features_b = ngram_builder.get_ngram_features('ngramngramngram')
        features_c = ngram_builder.get_ngram_features('rd')

        # features_a: UNK UNK UNK ord rdS PAD PAD PAD ... PAD
        # features_b: Sng ngr gra ram UNK UNK ngr gra ... UNK
        # features_c: UNK rdS PAD PAD PAD PAD PAD PAD ... PAD

        batched_features = ngram_builder.batchify([features_a, features_b, features_c])
        npt.assert_array_equal(batched_features.ngram_ids, np.array([[0, 0, 0, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                                     [2, 3, 4, 5, 0, 0, 3, 4, 5, 0, 0, 3, 4, 5, 0],
                                                                     [0, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
        npt.assert_array_equal(batched_features.ngrams_length, np.array([5, 15, 2]))
