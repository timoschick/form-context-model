from typing import List, Dict
import itertools
import numpy as np

import my_log

logger = my_log.get_logger('root')


class ContextFeatures:
    def __init__(self, contexts: List[List[str]], distances: List[List[int]]):
        self.contexts = contexts
        self.distances = distances
        self.number_of_contexts = len(contexts)
        self.words_per_context = [len(context) for context in contexts]

    def __repr__(self):
        return '{} , {}'.format(self.contexts, self.distances)


class BatchedContextFeatures:
    def __init__(self, vectors, distances, number_of_contexts, words_per_context):
        self.vectors = vectors
        self.distances = distances
        self.number_of_contexts = number_of_contexts
        self.words_per_context = words_per_context

    def __repr__(self):
        return '{}, {}, {}, {}'.format(self.vectors, self.distances, self.number_of_contexts, self.words_per_context)


class ContextBuilder:
    def __init__(self, word_embeddings: Dict[str, np.ndarray], vector_size: int, max_distance: int = 10):
        self.word_embeddings = word_embeddings
        self.vector_size = vector_size
        self.max_distance = max_distance

    def get_context_features(self, word: str, contexts: List[str]):

        if not isinstance(contexts, list):
            raise ValueError('A list of contexts must be passed to get_context_features, got {}'.format(contexts))

        contexts_as_lists = []  # type: List[List[str]]
        distances = []  # type: List[List[int]]

        for context in contexts:

            words = context.split()
            word_indices = [i for i, w in enumerate(words) if w == word]

            if not word_indices:
                raise ValueError('The word "{}" does not occur in the given context ("{}")'.format(word, context))

            context_with_distances = _add_distances(words, word_indices, self.max_distance)
            context_with_distances = [(i, w) for i, w in context_with_distances if
                                      w != word and w in self.word_embeddings]

            context = [w for i, w in context_with_distances]
            distance = [i for i, w in context_with_distances]

            if not context:
                continue

            contexts_as_lists.append(context)
            distances.append(distance)

        return ContextFeatures(contexts_as_lists, distances)

    def batchify(self, features: List[ContextFeatures]) -> BatchedContextFeatures:

        number_of_contexts = np.array([x.number_of_contexts for x in features], dtype=np.int32)
        max_context_length = np.amax(number_of_contexts, initial=0)

        words_per_context = np.array(
            list(itertools.zip_longest(*[x.words_per_context for x in features], fillvalue=0)),
            dtype=np.int32).T.reshape(len(features), max_context_length)

        max_words_per_context = np.amax(words_per_context, initial=0)

        vectors_shape = [len(features), max_context_length, max_words_per_context, self.vector_size]
        vectors = np.zeros(vectors_shape)

        for batch_idx, feature in enumerate(features):
            for context_idx, context in enumerate(feature.contexts):
                for word_idx, word in enumerate(context):
                    vectors[batch_idx][context_idx][word_idx] = self.word_embeddings[word]

        distances_shape = [len(features), max_context_length, max_words_per_context]
        distances = np.zeros(distances_shape)

        for batch_idx, feature in enumerate(features):
            for context_idx, context_distances in enumerate(feature.distances):
                for distance_idx, distance in enumerate(context_distances):
                    distances[batch_idx][context_idx][distance_idx] = distance

        return BatchedContextFeatures(vectors, distances, number_of_contexts, words_per_context)

def _add_distances(list, hit_indices, max_distance):
    list_with_distances = [(min([idx - hit_idx for hit_idx in hit_indices], key=abs), x) for idx, x in enumerate(list)]
    list_with_distances = [(_squeeze(idx, -max_distance, max_distance), x) for idx, x in list_with_distances]
    return list_with_distances


def _squeeze(number, min, max):
    if number < min:
        return min
    if number > max:
        return max
    return number
