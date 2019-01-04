"""
This file provides methods to evaluate our model on the Definitional Nonce dataset [1].

[1] Herbelot, A., and Baroni, M. 2017. High-risk learning: acquiring new word vectors
    from tiny data. In Proceedings of the 2017 Conference on Empirical Methods in
    Natural Language Processing, 304-309. Association for Computational Linguistics.

Note: this file is not executable, but its functions are called from form_context_model.py
"""

import numpy as np
import io
import re
import operator


class DefinitionalNonceEvaluator:
    def __init__(self, word2vec, eval_file, log_file):
        """
        Creates a new evaluator instance.
        :param word2vec: the original embedding vectors
        :param eval_file: the file in which the test words and sentences are stored, in the format of [1]
        """
        self.word2vec = word2vec
        self.test_data = []
        self.log_file = log_file

        if eval_file:
            with io.open(eval_file, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip() and not line.startswith('#'):
                        self.test_data.append(self._process_line(line))

        for datum in self.test_data:
            print(datum)

    def evaluate_model(self, model):
        vectors = []
        words = []

        for test_datum in self.test_data:
            word = test_datum['word']
            context = test_datum['context']
            vector = model.infer_vector(word=word, context=context)

            words.append(word)
            vectors.append(vector)

        stats = self.stats(vectors, words)

        self.log_file.write('MRR:' + str(stats['mean reciprocal rank']) + '\tMedian rank: ' + str(
            stats['median']) + '\tStats:' + str(stats))
        self.log_file.write('\n')

    def _process_line(self, line):

        # example for a valid line: pride	___ is an inwardly directed emotion that carries two common meanings
        comps = re.split(r'\t+', line)

        word = comps[0]
        sentence = comps[1]

        return {'word': word, 'context': sentence}

    def stats(self, vectors, words):
        """Computes the Mean Reciprocal Rank and Median Rank given a list of ``vectors`` and ``words``"""
        assert len(vectors) == len(words)
        ranks = []
        word_ranks = {}
        closest = {}
        count = 0

        for (vector, word) in zip(vectors, words):
            closest[word] = get_closest(vector, self.word2vec)

            rank = compute_rank(vector, word, self.word2vec)
            ranks.append(rank)
            word_ranks[word] = rank
            count += 1
            if count % 10 == 0:
                mrr = get_mean_reciprocal_rank(ranks)
                median = np.median(ranks)
                print('Done computing ranks for', count, 'of', len(vectors), 'pairs, mrr =', mrr, 'median =', median)

        mrr = get_mean_reciprocal_rank(ranks)
        median = np.median(ranks)

        sorted_words_asc = sorted(word_ranks.items(), key=operator.itemgetter(1))
        return {'mean reciprocal rank': mrr, 'median': median, 'words': sorted_words_asc, 'closest': closest}


def compute_rank(inferred_embedding, word, base_embeddings):
    closer_words = get_closer_than(inferred_embedding, word, base_embeddings)
    return len(closer_words) + 1


def get_closer_than(inferred_embedding, word, base_embeddings):
    """Returns all entities from base_embeddings that are closer to `inferred_embedding` than the actual `word`"""
    all_distances = base_embeddings.distances(inferred_embedding)
    word_index = base_embeddings.vocab[word].index
    closer_node_indices = np.where(all_distances < all_distances[word_index])[0]
    return [base_embeddings.index2entity[index] for index in closer_node_indices if index != word_index]


def get_closest(inferred_embedding, base_embeddings, num=10):
    """Returns the `num` closest entities from base_embeddings to `inferred_embedding`"""
    all_distances = base_embeddings.distances(inferred_embedding)
    closest_word_indices = np.argsort(all_distances)[:num]
    return [base_embeddings.index2entity[index] for index in closest_word_indices]


def get_mean_reciprocal_rank(ranks):
    """
    Computes the MRR given a collection of ranks.
    :param ranks: the collection of ranks
    :return: the mean reciprocal rank
    """
    ranks_inverted = [1 / x for x in ranks]
    return 1 / len(ranks) * sum(ranks_inverted)
