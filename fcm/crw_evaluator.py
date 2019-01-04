import numpy as np
import io
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

FLOAT = np.float32
DATASETS = ['test', 'dev']


class ContextualizedRareWordsEvaluator:
    def __init__(self, log_file, dataset='test'):

        np.random.seed(42)

        assert dataset in DATASETS
        self.log_file = log_file
        self.dataset = dataset

    def evaluate_model(self, model):

        if self.dataset == 'dev':
            eval_baselines(model, trials=1, maxf=7, log_file=self.log_file, dir='../data/crw/dev/',
                           scorefile='CRW-550.txt')

        else:
            eval_baselines(model, trials=10, log_file=self.log_file) # TODO trials=100 for final results


def cosine(v1, v2):
    '''computes cosine similarity between rows
    Args:
      v1: 2-d numpy array
      v2: 2-d numpy array
    Returns:
      1-d numpy array with cosine similarities
    '''
    assert (v1.shape == v2.shape)
    v1 = normalize(v1)
    v2 = normalize(v2)
    return (v1 * v2).sum(axis=1)


def eval_crw(w2v, rarew2v, scorefile='CRW-562.txt', dir=None, context_words=None):
    '''evaluates the given embeddings on the CRW dataset
    Args:
      w2v: {string: numpy array}; embeddings of non-rare words
      rarew2v: {string: numpy array}; embeddings of rare words
    Returns:
      float; Spearman correlation coefficient for the given embeddings
    '''
    dim = w2v['word'].shape[0]
    z = np.zeros(dim)
    with io.open(dir + scorefile, 'r', encoding='utf-8') as f:
        lines = list(f)

    pairs = [l.split()[:2] for l in lines]
    scores = [float(l.split()[-1]) for l in lines]

    predscores = cosine(np.vstack([w2v[p[0]] if p[0] in w2v.keys() else rarew2v.get(p[0], z) for p in pairs]),
                        np.vstack([rarew2v.get(p[1], z) for p in pairs]))

    return spearmanr(scores, predscores)[0]


def get_context_words(rarevocab, dir):
    contexts = {}
    for word in rarevocab:
        with io.open(dir + 'context/' + word + '.txt', 'r', encoding='utf-8') as f:
            contexts[word] = [get_context_from_line(line, word) for line in f]

    return contexts


def get_context_from_line(line, rareword):
    words = line.lower().split()
    return words


def get_oov_vector(word, contexts, model):
    # contexts is a list of lists of words, where each list also contains "word" once
    context_strings = []
    for context in contexts:
        context_strings.append(' '.join(context))

    #context = ' '.join(context_strings)
    vector = model.infer_vector(word=word, context=context_strings)

    return vector


def print_scores(methods, scores, maxf, log_file):
    '''prints Spearman coefficient average and standard deviation
    Args:
      methods: [string]; name of all methods
      scores: 3-D numpy array; the axes correspond to (method, frequency, trial)
    Returns:
    '''
    mean = scores.mean(axis=0)
    std = scores.std(axis=0)
    print('Average Spearman correlation coefficient')
    log_file.write('Average Spearman correlation coefficient\n')

    print('freq\t' + '\t'.join(methods))
    log_file.write('freq\t' + '\t'.join(methods) + '\n')
    for j in range(maxf):
        print(str(2 ** j) + '\t' + '\t'.join(['%.4f' % x for x in mean[:, j]]))
        log_file.write(str(2 ** j) + '\t' + '\t'.join(['%.4f' % x for x in mean[:, j]]) + '\n')
    print('Standard Deviation')
    log_file.write('Standard Deviation\n')
    print('freq\t' + '\t'.join(methods))
    log_file.write('freq\t' + '\t'.join(methods) + '\n')
    for j in range(maxf):
        log_file.write(str(2 ** j) + '\t' + '\t'.join(['%.4f' % x for x in std[:, j]]))
        print(str(2 ** j) + '\t' + '\t'.join(['%.4f' % x for x in std[:, j]]))

    log_file.write('\n')
    log_file.flush()


def eval_baselines(model, trials=100, maxf=8, log_file=None, rarevocab=None, context_words=None,
                   scorefile='CRW-562.txt', dir='../data/crw/'):
    '''
    evaluates the form-context-model on the CRW dataset.
    '''

    if rarevocab is None:
        with io.open(dir + 'rarevocab.txt', 'r', encoding='utf-8') as f:
            rarevocab = sorted([line.split()[0] for line in f])

        print('rare vocab loaded')

    else:
        rarevocab = [w for w in rarevocab if len(context_words[w]) >= 2 ** maxf]

    if context_words is None:
        context_words = get_context_words(rarevocab, dir)

    # Run trials
    methods = ['mine']
    scores = np.zeros(shape=(trials, 1, maxf))
    for trial in range(trials):

        perm = np.random.permutation(2 ** maxf - 1)

        for logfreq in range(maxf):
            freq = 2 ** logfreq
            # Test Average vectors.
            print('freq: ', freq)

            avgw2v = {word: get_oov_vector(word, [context_words[word][perm[i]] for i in range(freq - 1, 2 * freq - 1)],
                                           model=model) for word in rarevocab}
            scores[trial][0][logfreq] = eval_crw(model.batch_builder.word2vec, avgw2v, dir=dir,
                                                 scorefile=scorefile, context_words=context_words)
    print_scores(methods, scores, maxf, log_file)
