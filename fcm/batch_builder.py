"""
This file contains the logic required to obtain batches of training instances (w,C) for
training the form-context model.
"""
from collections import Counter

import numpy as np
import random
import io
import time
import itertools
import re
import logging

from ngram_builder import to_n_gram

logger = logging.getLogger(__name__)

UNK_TOKEN = 'UNK'
PAD_TOKEN = 'PAD'

UNK_ID = 0
PAD_ID = 1


class OOVBatchBuilder:
    def __init__(self,
                 word2vec,
                 test_words_file,
                 data_files=None,
                 unigram_file=None,
                 letter_threshold=100,
                 emb_dim=400,
                 sample_sentences=None,
                 ngram_range=None,
                 min_word_count=100,
                 max_copies=5,
                 ngram_threshold=3,
                 ngram_dropout=0
                 ):

        random.seed(1234)

        self.data_files = data_files

        self.min_word_count = min_word_count
        self.max_copies = max_copies

        self.emb_dim = emb_dim
        self.word2vec = word2vec

        self.ngram_range = ngram_range
        self.sample_sentences = sample_sentences

        self.ngram_threshold = ngram_threshold
        self.ngram_dropout = ngram_dropout

        self.generate_batch_time = 0
        self.fill_buffers_time = 0

        self.unigram_counts = []
        self.word2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id2word = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

        self.ngram2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id2letter = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

        self.total_word_count = 0
        self.nr_of_words = 2

        self.total_letter_count = 0
        self.nr_of_letters = 2

        self.zeroes = np.zeros(self.emb_dim)
        letter_counts = Counter()

        self.word_context_dict = {}
        self.used_sentences = {}

        self.forbidden_words = set()
        duplicate_words = []

        if isinstance(test_words_file, list):
            files = test_words_file
        elif test_words_file is None:
            files = []
        else:
            files = [test_words_file]

        for f in files:
            if f is not None:
                with io.open(f, "r", encoding="utf-8") as file:
                    for line in file:
                        if line.strip() and not line.startswith('#'):
                            blacklist_word = line.split()[0]
                            if blacklist_word in self.forbidden_words:
                                duplicate_words.append(blacklist_word)
                            else:
                                self.forbidden_words.add(blacklist_word)

            if duplicate_words:
                logger.warn('Found %d duplicate blacklist words, the first 10 being: %s',
                            len(duplicate_words), duplicate_words[:10])
            logger.info('Number of unique blacklist words: %d', len(self.forbidden_words))
            logger.debug('First 10 blacklist words: %s', [w for w in self.forbidden_words][:10])

        with io.open(unigram_file, "r", encoding="utf-8") as file:

            line_index = 0

            for line in file:

                comps = line.split()

                if len(comps) != 2:
                    raise ValueError("Line " + line + " does not follow the required format")

                word = comps[0]
                count = int(comps[1])

                if word in self.forbidden_words:
                    continue

                if line_index == UNK_ID and word != UNK_TOKEN:
                    raise ValueError("Entry ", UNK_ID, " of the vocab file must be ", UNK_TOKEN)

                if line_index == PAD_ID and word != PAD_TOKEN:
                    raise ValueError("Entry ", PAD_ID, " of the vocab file must be ", PAD_TOKEN)

                line_index += 1

                self.unigram_counts.append(count)

                if word not in [PAD_TOKEN, UNK_TOKEN]:

                    if word in self.word2id.keys():
                        logger.warn('Duplicate word in vocab: %s', word)

                    self.word2id[word] = self.nr_of_words
                    self.id2word[self.nr_of_words] = word
                    self.nr_of_words += 1

                    if not self.ngram_range:
                        letter_counts.update(list(word))

                    else:
                        self.nr_of_letters = 0
                        letter_threshold = self.ngram_threshold
                        letter_counts.update(to_n_gram(word, self.ngram_range))

                self.total_word_count += int(comps[1])

        for (letter, count) in sorted(letter_counts.items()):
            if count > letter_threshold:
                self.ngram2id[letter] = self.nr_of_letters
                self.id2letter[self.nr_of_letters] = letter
                self.nr_of_letters += 1

        self.processed_words = 0

        logger.info('Nr of words: %d', self.nr_of_words)
        logger.info('Nr of ngrams: %d', len(self.ngram2id))

        self.buffer = None
        self.sequence = None

        self.line_count = 0
        self.time = time.time()

        self.passes_through_dataset = 0
        self.reached_end_of_dataset = False

        if self.data_files is not None:
            self.reset()

    def reset(self):

        self.passes_through_dataset += 1
        self.buffer = []
        self.sequence = []

        for word in sorted(self.word2id.keys()):

            word_id = self.word2id[word]
            unigram_count = self.unigram_counts[word_id]

            if len(word) >= 3 and unigram_count >= self.min_word_count:
                thresholded_count = int(max(1, min(unigram_count / self.min_word_count, self.max_copies)))
                self.sequence += [word] * thresholded_count

        random.shuffle(self.sequence)

        count = 0

        for data_file in self.data_files:
            file = io.open(data_file, "r", encoding="utf-8")
            count += 1
            logger.info('Reading file %s (%d of %d)', data_file, count, len(self.data_files))
            for line in file:
                comps = re.split(r'\t', line)
                word = comps[0]

                if word in self.forbidden_words:
                    continue

                if len(comps) == 2 and comps[1] == '\n':
                    continue

                remaining_sents = random.randint(self.sample_sentences[0], self.sample_sentences[1])
                index_choice = []

                while remaining_sents > 0:
                    if word in self.used_sentences:
                        indices = set(range(1, len(comps)))
                        unused_indices = list(indices - self.used_sentences[word])

                    else:
                        unused_indices = list(range(1, len(comps)))

                    # select self.sample_sentences sentences from the unused indices
                    random.shuffle(unused_indices)
                    new_indices = unused_indices[:remaining_sents]

                    if len(unused_indices) >= remaining_sents:
                        index_choice += new_indices
                        remaining_sents = 0

                    else:
                        index_choice += unused_indices
                        remaining_sents -= len(unused_indices)

                    if word in self.used_sentences:
                        self.used_sentences[word] |= set(new_indices)
                    else:
                        self.used_sentences[word] = set(new_indices)

                    if len(self.used_sentences[word]) == len(comps) - 1:
                        self.used_sentences[word].clear()

                sentence_choice = [comps[i] for i in index_choice]
                self.word_context_dict[word] = sentence_choice

        self.line_count = 0
        self.time = time.time()
        self.reached_end_of_dataset = False

    def _fill_buffer(self, print_every_n_steps=500):

        if len(self.sequence) <= self.line_count:
            self.reached_end_of_dataset = True
            return

        word = self.sequence[self.line_count]
        self.line_count += 1

        if self.line_count % print_every_n_steps == 0:
            logger.info('Done processing %d lines (%.2f%% of iteration %d)', self.line_count,
                        100 * self.line_count / len(self.sequence), self.passes_through_dataset)

        if word not in self.word_context_dict:
            # print('word', current_word, 'is not in the dictionary')
            return

        sentences = self.word_context_dict[word]

        if word not in self.word2vec.vocab or word in self.forbidden_words:
            return

        contexts, distances = self.sentences_to_contexts(word, sentences, blacklist_words=self.forbidden_words)
        total_context_length = sum([len(c) for c in contexts])

        if total_context_length < 5:
            return

        context_vectors = [[self.word2vec[word] for word in context] for context in contexts]
        context_length = len(contexts)
        words_per_context = [len(context) for context in contexts]

        target = self.word2vec[word]

        letters = to_n_gram(word, ngram_range=self.ngram_range, dropout_probability=self.ngram_dropout)
        letters = [self.ngram2id[letter] if letter in self.ngram2id else UNK_ID for letter in letters]

        if len(letters) == 0:
            return

        training_instance = {
            'letters': letters,
            'word_length': len(letters),
            'context_vectors': context_vectors,
            'context_length': context_length,
            'distances': distances,
            'words_per_context': words_per_context,
            'target': target
        }

        self.buffer.append(training_instance)

    def generate_batch_from_buffer(self, batch_size):

        if len(self.buffer) < batch_size:
            while len(self.buffer) < 50 * batch_size and not self.reached_end_of_dataset:
                self._fill_buffer()

        if len(self.buffer) < batch_size:
            raise EndOfDatasetException()

        words = np.array(
            list(itertools.zip_longest(*self._buffer_entries(batch_size, 'letters'), fillvalue=PAD_ID)),
            dtype=np.int32).T
        word_lengths = np.array(self._buffer_entries(batch_size, 'word_length'), dtype=np.int32)
        context_lengths = np.array(self._buffer_entries(batch_size, 'context_length'), dtype=np.int32)

        words_per_context = np.array(
            list(itertools.zip_longest(*self._buffer_entries(batch_size, 'words_per_context'), fillvalue=0)),
            dtype=np.int32).T

        max_context_length = np.amax(context_lengths)
        max_words_per_context = np.amax(words_per_context)

        context_vectors_shape = [batch_size, max_context_length, max_words_per_context, self.emb_dim]
        context_vectors = np.zeros(context_vectors_shape)

        batch_index = 0
        for example_cvs in self._buffer_entries(batch_size, 'context_vectors'):
            context_index = 0
            for context in example_cvs:
                word_index = 0
                for word_vector in context:
                    context_vectors[batch_index][context_index][word_index] = word_vector
                    word_index += 1
                context_index += 1
            batch_index += 1

        distance_vectors_shape = [batch_size, max_context_length, max_words_per_context]
        distance_vectors = np.zeros(distance_vectors_shape)

        batch_index = 0
        for example_dsts in self._buffer_entries(batch_size, 'distances'):
            context_index = 0
            for distances in example_dsts:
                word_index = 0
                for dist in distances:
                    distance_vectors[batch_index][context_index][word_index] = dist
                    word_index += 1
                context_index += 1
            batch_index += 1

        target = self._buffer_entries(batch_size, 'target')

        del self.buffer[:batch_size]

        return {
                   'ngrams': words,
                   'ngram_lengths': word_lengths,
                   'context_vectors': context_vectors,
                   'context_lengths': context_lengths,
                   'words_per_context': words_per_context,
                   'distances': distance_vectors
               }, target

    def generate_batch_from_context(self, word, sentences):

        ngrams = to_n_gram(word, ngram_range=self.ngram_range)
        ngram_ids = np.array([[self.ngram2id[ngram] if ngram in self.ngram2id else UNK_ID for ngram in ngrams]])
        ngram_lengths = np.array([len(ngrams)])

        contexts, distances = self.sentences_to_contexts(word, sentences, blacklist_words=['___'])

        context_lengths = np.array([len(contexts)])
        words_per_context = np.array([[len(context) for context in contexts]])

        max_context_length = np.amax(context_lengths) if context_lengths.size > 0 else 0
        max_words_per_context = np.amax(words_per_context) if words_per_context.size > 0 else 0

        context_vectors_shape = [1, max_context_length, max_words_per_context, self.emb_dim]
        context_vectors = np.zeros(context_vectors_shape)

        distance_vectors_shape = [1, max_context_length, max_words_per_context]
        distance_vectors = np.zeros(distance_vectors_shape)

        context_index = 0
        for context in contexts:
            word_index = 0
            for w in context:
                context_vectors[0][context_index][word_index] = self.word2vec[w]
                word_index += 1
            context_index += 1

        context_index = 0
        for distance in distances:
            word_index = 0
            for dist in distance:
                distance_vectors[0][context_index][word_index] = dist
                word_index += 1
            context_index += 1

        target = np.zeros([1, self.emb_dim])

        return {'ngrams': ngram_ids,
                'ngram_lengths': ngram_lengths,
                'context_vectors': context_vectors,
                'context_lengths': context_lengths,
                'words_per_context': words_per_context,
                'distances': distance_vectors
                }, target

    def _buffer_entries(self, batch_size, name):
        return [x[name] for x in self.buffer[:batch_size]]

    def sentences_to_contexts(self, word, sentences, blacklist_words=None, multiple_contexts=True, max_distance=10):
        if not isinstance(sentences, list):
            if isinstance(sentences, str):
                sentences = re.split(r'\t', sentences)
            else:
                raise TypeError("Expected list of strings or tab-separated strings")

        if blacklist_words is None:
            blacklist_words = []
        contexts = []
        distances = []

        for sentence in sentences:

            words = sentence.split()
            word_indices = [i for i, w in enumerate(words) if w == word or w == '___']

            context_with_distances = [(min([i - wi for wi in word_indices], key=abs), w)
                                      for i, w in enumerate(words)]

            context_with_distances = [(min(max(i, -max_distance), max_distance) + max_distance, w)
                                      for i, w in context_with_distances if
                                      w not in blacklist_words and
                                      w != word and
                                      w in self.word2vec.vocab]

            context = [w for i, w in context_with_distances]
            distance = [i for i, w in context_with_distances]

            if not context:
                continue
            elif multiple_contexts:
                contexts.append(context)
                distances.append(distance)
            else:
                contexts += context
                distances += distance

        return contexts, distances


def load_builder(config):
    if config['base_embeddings']['format'] == 'text':
        from embedding_loader import get_w2v_model_file
        word2vec = get_w2v_model_file(filename=config['base_embeddings']['file'])
    else:
        from embedding_loader import get_w2v_model_gensim
        word2vec = get_w2v_model_gensim(filename=config['base_embeddings']['file'])

    batch_builder = OOVBatchBuilder(
        word2vec=word2vec,
        unigram_file=config['vocab_file'],
        test_words_file=config['word_blacklist_file'],
        ngram_range=[config['nmin'], config['nmax']],
        emb_dim=config['emb_dim'])

    return batch_builder


class EndOfDatasetException(Exception):
    pass
