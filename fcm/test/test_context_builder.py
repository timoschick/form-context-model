import unittest
import numpy as np
import numpy.testing as npt

from context_builder import ContextBuilder, _add_distances

import my_log

logger = my_log.get_logger('root')


class TestContextBuilder(unittest.TestCase):

    def test_context_builder_empty_context(self):

        embeddings = {
            'this': np.array([1, 2, 3]),
            'is': np.array([4, 5, 6]),
            'word': np.array([7, 8, 9])
        }

        context_builder = ContextBuilder(embeddings, vector_size=3, max_distance=2)
        context_features = context_builder.get_context_features('word2', ['context for word2 with only unk words'])
        batched_context_features = context_builder.batchify([context_features])

        logger.info(batched_context_features.words_per_context.shape)
        logger.info(batched_context_features.distances.shape)
        logger.info(batched_context_features.vectors.shape)
        logger.info(batched_context_features.number_of_contexts.shape)

    def test_context_builder(self):
        embeddings = {
            'this': np.array([1, 2, 3]),
            'is': np.array([4, 5, 6]),
            'word': np.array([7, 8, 9])
        }

        e_this = [1, 2, 3]
        e_is = [4, 5, 6]
        e_word = [7, 8, 9]
        e_null = [0, 0, 0]

        context_builder = ContextBuilder(embeddings, vector_size=3, max_distance=2)

        context_features = context_builder.get_context_features('word',
                                                                ['this is the word .', 'the word is this is is is this'])

        self.assertEquals(context_features.contexts, [['this', 'is'], ['is', 'this', 'is', 'is', 'is', 'this']])
        self.assertEquals(context_features.distances, [[-2, -2], [1, 2, 2, 2, 2, 2]])

        context_features = context_builder.get_context_features('word',
                                                                ['word this is word the word word word is word'])
        self.assertEquals(context_features.contexts, [['this', 'is', 'is']])

        context_builder = ContextBuilder(embeddings, vector_size=3, max_distance=5)

        context_a_features = context_builder.get_context_features('word',
                                                                  ['this is the word',
                                                                   'the word is this is is is this'])

        self.assertEquals(context_a_features.contexts, [['this', 'is'], ['is', 'this', 'is', 'is', 'is', 'this']])
        self.assertEquals(context_a_features.distances, [[-3, -2], [1, 2, 3, 4, 5, 5]])

        context_b_features = context_builder.get_context_features('this', ['this is', 'is this', 'this word'])

        batch_features = context_builder.batchify([context_a_features, context_b_features])

        npt.assert_equal(batch_features.number_of_contexts, np.array([2, 3]))
        npt.assert_equal(batch_features.words_per_context, np.array([[2, 6, 0], [1, 1, 1]]))

        npt.assert_equal(batch_features.distances, np.array(
            [
                [[-3, -2, 0, 0, 0, 0], [1, 2, 3, 4, 5, 5], [0, 0, 0, 0, 0, 0]],
                [[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
            ]
        ))

        npt.assert_equal(batch_features.vectors, np.array(
            [
                [[e_this, e_is, e_null, e_null, e_null, e_null],
                 [e_is, e_this, e_is, e_is, e_is, e_this],
                 [e_null, e_null, e_null, e_null, e_null, e_null]],

                [[e_is, e_null, e_null, e_null, e_null, e_null],
                 [e_is, e_null, e_null, e_null, e_null, e_null],
                 [e_word, e_null, e_null, e_null, e_null, e_null]]
            ]
        ))

    def test_add_distances(self):
        self.assertEquals(_add_distances(['a', 'b', 'c', 'd'], [1], 10),
                          [(-1, 'a'), (0, 'b'), (1, 'c'), (2, 'd')])
        self.assertEquals(_add_distances(['a', 'b', 'b', 'c', 'c', 'c', 'd'], [1, 5], 10),
                          [(-1, 'a'), (0, 'b'), (1, 'b'), (2, 'c'), (-1, 'c'), (0, 'c'), (1, 'd')])
        self.assertEquals(_add_distances(['a', 'b', 'b', 'c', 'c', 'c', 'd'], [1], 10),
                          [(-1, 'a'), (0, 'b'), (1, 'b'), (2, 'c'), (3, 'c'), (4, 'c'), (5, 'd')])
        self.assertEquals(_add_distances(['a', 'b', 'b', 'c', 'c', 'c', 'd'], [1], 3),
                          [(-1, 'a'), (0, 'b'), (1, 'b'), (2, 'c'), (3, 'c'), (3, 'c'), (3, 'd')])
        self.assertEquals(_add_distances(['a', 'b', 'b', 'c', 'c', 'c', 'd'], [6], 3),
                          [(-3, 'a'), (-3, 'b'), (-3, 'b'), (-3, 'c'), (-2, 'c'), (-1, 'c'), (0, 'd')])
