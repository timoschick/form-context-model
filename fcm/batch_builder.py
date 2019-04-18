from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import random
import re
import jsonpickle

import my_log
import utils

from context_builder import ContextBuilder, BatchedContextFeatures
from context_builder import ContextFeatures
from ngram_builder import NGramBuilder, BatchedNGramFeatures
from ngram_builder import NGramFeatures

logger = my_log.get_logger('root')


class ProcessedInput:
    def __init__(self, ngram_features: NGramFeatures, context_features: ContextFeatures, target: np.ndarray):
        self.ngram_features = ngram_features
        self.context_features = context_features
        self.target = target


class EndOfDatasetException(Exception):
    pass


class AbstractInputProcessor(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def generate_batch_from_buffer(self, batch_size: int) -> List[ProcessedInput]:
        pass

    @abstractmethod
    def generate_batch_from_input(self, word: str, contexts: List[str]) -> List[ProcessedInput]:
        pass


class InputProcessor(AbstractInputProcessor):
    def __init__(self, word_embeddings_file: str, word_embeddings_format: str, train_files: List[str], vocab_file: str,
                 vector_size: int, ngram_threshold: int = 4, nmin: int = 3, nmax: int = 5, ngram_dropout: float = 0,
                 max_distance: int = 10, min_word_count: int = 100, max_copies: int = 5, smin: int = 20,
                 smax: int = 20, keep_punctuation=False):
        random.seed(1234)

        self.word_embeddings_file = word_embeddings_file
        self.word_embeddings_format = word_embeddings_format
        self.train_files = train_files
        self.vocab_file = vocab_file

        self.vector_size = vector_size
        self.ngram_threshold = ngram_threshold
        self.nmin = nmin
        self.nmax = nmax
        self.ngram_dropout = ngram_dropout
        self.max_distance = max_distance
        self.min_word_count = min_word_count
        self.max_copies = max_copies
        self.smin = smin
        self.smax = smax
        self.keep_punctuation = keep_punctuation

        self.word_counts = None  # type: Dict[str,int]
        self.word_embeddings = None  # type: Dict[str, np.ndarray]
        self.ngram_builder = None  # type: NGramBuilder
        self.context_builder = None  # type: ContextBuilder
        self._setup()

        self.buffer = None  # type: List[ProcessedInput]
        self.train_file_idx = 0
        self.reset()

    def _setup(self):

        self.word_counts = {}

        with open(self.vocab_file, 'r', encoding='utf8') as file:
            for line in file:
                word, count = line.split()
                self.word_counts[word] = int(count)

        self.word_embeddings = utils.load_embeddings(self.word_embeddings_file, self.word_embeddings_format, self.keep_punctuation)
        self.ngram_builder = NGramBuilder(self.vocab_file, ngram_threshold=self.ngram_threshold,
                                          nmin=self.nmin, nmax=self.nmax)

        self.context_builder = ContextBuilder(self.word_embeddings, vector_size=self.vector_size,
                                              max_distance=self.max_distance)

    def reset(self) -> None:
        random.shuffle(self.train_files)
        self.buffer = []
        self.train_file_idx = 0

    def generate_batch_from_buffer(self, batch_size: int) -> List[ProcessedInput]:

        if len(self.buffer) < batch_size:
            while self._fill_buffer():
                continue

        if len(self.buffer) < batch_size:
            raise EndOfDatasetException()

        batch = self.buffer[:batch_size]
        batched_ngram_features = self.ngram_builder.batchify([x.ngram_features for x in batch])
        batched_context_features = self.context_builder.batchify([x.context_features for x in batch])
        batched_targets = [x.target for x in batch]
        del (self.buffer[:batch_size])

        return _convert_to_tf_input(batched_ngram_features, batched_context_features), batched_targets

    def generate_batch_from_input(self, word: str, contexts: List[str]) -> List[ProcessedInput]:
        ngram_features = self.ngram_builder.get_ngram_features(word)
        context_features = self.context_builder.get_context_features(word, contexts)

        batched_ngram_features = self.ngram_builder.batchify([ngram_features])
        batched_context_features = self.context_builder.batchify([context_features])
        target = np.zeros([1, self.vector_size])

        return _convert_to_tf_input(batched_ngram_features, batched_context_features), target

    def _get_occurrences(self, word):
        if word not in self.word_counts or word not in self.word_embeddings:
            return 0
        word_count = self.word_counts[word]
        return int(max(1, min(word_count / self.min_word_count, self.max_copies)))

    def _fill_buffer(self) -> bool:
        # select the next train file
        if self.train_file_idx == len(self.train_files):
            logger.info('Reached the end of the dataset')
            return False

        logger.info('Processing training file {} of {}'.format(self.train_file_idx + 1, len(self.train_files)))
        train_file = self.train_files[self.train_file_idx]
        self.train_file_idx += 1
        self._fill_buffer_from_file(train_file)
        random.shuffle(self.buffer)
        logger.info('Done processing training file, batch size is {}'.format(len(self.buffer)))

    def _fill_buffer_from_file(self, file):
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                self._fill_buffer_from_line(line)

    def _fill_buffer_from_line(self, line):
        comps = re.split(r'\t', line)
        word = comps[0]
        all_contexts = comps[1:]
        random.shuffle(all_contexts)

        if len(comps) == 2 and comps[1] == '\n':
            return

        occurrences = self._get_occurrences(word)
        for _ in range(occurrences):
            number_of_contexts = random.randint(self.smin, self.smax)
            contexts = all_contexts[:number_of_contexts]
            self._fill_buffer_from_contexts(word, contexts)
            del (all_contexts[:number_of_contexts])

    def _fill_buffer_from_contexts(self, word: str, contexts: List[str]):
        ngram_features = self.ngram_builder.get_ngram_features(word, dropout_probability=self.ngram_dropout)
        context_features = self.context_builder.get_context_features(word, contexts)
        target = self.word_embeddings[word]
        if not contexts:
            return
        self.buffer.append(ProcessedInput(ngram_features, context_features, target))

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['word_counts']
        del odict['word_embeddings']
        del odict['ngram_builder']
        del odict['context_builder']
        del odict['buffer']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self._setup()

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf8') as f:
            f.write(jsonpickle.encode(self))

    @classmethod
    def load(cls, path: str) -> 'InputProcessor':
        with open(path, 'r', encoding='utf8') as f:
            batch_builder = jsonpickle.decode(f.read())
            batch_builder._setup()
        return batch_builder


def _convert_to_tf_input(ngram_features: BatchedNGramFeatures, context_features: BatchedContextFeatures) \
        -> Dict[str, np.ndarray]:
    return {'ngrams': ngram_features.ngram_ids,
            'ngram_lengths': ngram_features.ngrams_length,
            'context_vectors': context_features.vectors,
            'context_lengths': context_features.number_of_contexts,
            'words_per_context': context_features.words_per_context,
            'distances': context_features.distances
            }
