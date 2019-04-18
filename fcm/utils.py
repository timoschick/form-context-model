import io
import random
import re, string
import numpy as np

from gensim.models import Word2Vec
import my_log

logger = my_log.get_logger('root')

PUNCTUATION_PATTERN = re.compile("^[{}]+$".format(re.escape(string.punctuation)))
START_SYMBOL = '<S>'
random.seed(1234)


def to_n_gram(word, ngram_range, dropout_probability=0):
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


def load_embeddings(file_name: str, format: str, keep_punctuation: bool = False):
    if format == 'text':
        return get_w2v_model_file(filename=file_name, keep_punctuation=keep_punctuation)
    else:
        return get_w2v_model_gensim(filename=file_name, keep_punctuation=keep_punctuation)


def get_w2v_model_gensim(filename, keep_punctuation: bool = False):
    if not keep_punctuation:
        raise NotImplementedError('get_w2v_model_gensim is only implemented with keep_punctuation = True')
    logger.info('Loading embeddings from %s', filename)
    w2v_model = Word2Vec.load(filename)
    word2vec = w2v_model.wv
    del w2v_model
    logger.info('Done loading embeddings')
    return word2vec


def get_w2v_model_file(filename, keep_punctuation: bool = False):
    logger.info('Loading embeddings from %s', filename)
    w2v = DummyDict()
    w2v.update({w: v for w, v in _load_vectors(filename, skip=True, keep_punctuation=keep_punctuation)})
    w2v.vocab = w2v.keys()
    logger.info('Done loading embeddings')
    return w2v


def _load_vectors(path, skip=False, keep_punctuation: bool = False):
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if skip:
                skip = False
            else:
                index = line.index(' ')
                word = line[:index]
                if not keep_punctuation and _is_punctuation(word):
                    continue
                yield word, np.array([np.float(entry) for entry in line[index + 1:].split()])


def _is_punctuation(word: str):
    return PUNCTUATION_PATTERN.match(word)


class DummyDict(dict):
    pass
