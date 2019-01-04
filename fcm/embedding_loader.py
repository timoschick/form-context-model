import io
import numpy as np
from gensim.models import Word2Vec
import logging

logger = logging.getLogger(__name__)

def get_w2v_model_gensim(filename='../data/oov/wiki_all.sent.split.model'):
    logger.info('Loading embeddings from %s', filename)
    w2v_model = Word2Vec.load(filename)
    word2vec = w2v_model.wv
    del w2v_model
    logger.info('Done loading embeddings')
    return word2vec


def get_w2v_model_file(filename='../data/crw/vectors.txt', dim=300):
    logger.info('Loading embeddings from %s', filename)
    w2v = MyDict()
    w2v.update({w: v for w, v in _load_vectors(filename, dim=dim, skip=True)})
    w2v.vocab = w2v.keys()
    logger.info('Done loading embeddings')
    return w2v


def _load_vectors(path, dim=300, skip=False):
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if skip:
                skip = False
            else:
                index = line.index(' ')
                word = line[:index]
                yield word, np.array([np.float(entry) for entry in line[index + 1:].split()[:dim]])


class MyDict(dict):
    pass
