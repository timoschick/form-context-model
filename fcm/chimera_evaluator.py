from scipy import stats
from numpy import *
import math
import io
from sklearn.preprocessing import normalize


class ChimeraEvaluator:
    def __init__(self, word2vec, chimeras_dir, train=False, log_file=None):
        self.word2vec = word2vec
        self.chimeras_dir = chimeras_dir
        self.train = train
        self.log_file = log_file

    def evaluate_model(self, model):

        suffix = 'train' if self.train else 'test'

        files = [self.chimeras_dir + '/' +
                 'chimeras.dataset.l' + str(i) + '.tokenised.' + suffix + '.txt'
                 for i in [2, 4, 6]]

        for file in files:
            self.eval(model, file)

    def eval(self, model, dataset):
        spearmans = []

        print('evaluating file ' + dataset)
        if self.log_file:
            self.log_file.write('evaluating file ' + dataset + '\n')

        nonce = "___"
        c = 0
        f = io.open(dataset, encoding='utf-8')
        for l in f:
            fields = l.rstrip('\n').split('\t')
            sentences = []
            for s in fields[1].split("@@"):

                if nonce not in s:
                    s = nonce + ' ' + s

                sentences.append(s)
            probes = fields[2].split(',')
            responses = fields[3].split(',')

            system_responses = []
            human_responses = []
            p_count = 0


            # get the vector for the nonce
            nonce_vec, ms = model.infer_vector(nonce, sentences)
            set_printoptions(threshold=nan)

            for sentence in sentences:
                print(sentence.encode('utf-8'))

            print('ms=' + str(ms))

            for p in probes:

                if p not in self.word2vec:
                    p_vec = zeros(model.config['emb_dim'])
                else:
                    p_vec = self.word2vec[p]

                #try:
                cos = cosine(nonce_vec, p_vec)
                system_responses.append(cos)
                human_responses.append(responses[p_count])
                #except:
                #    print("ERROR processing", p)
                p_count += 1

            if len(system_responses) > 1:
                sp = spearman(human_responses, system_responses)
                if not math.isnan(sp):
                    spearmans.append(sp)
                    print(sp)
                else:
                    print('NAN sp, sys_rep are' + str(system_responses))
            c += 1
            print('----------------')

        f.close()

        print("AVERAGE RHO:", float(sum(spearmans)) / float(len(spearmans)))
        if self.log_file:
            self.log_file.write("AVERAGE RHO:" + str(float(sum(spearmans)) / float(len(spearmans))) + '\n')


def spearman(x, y):
    return stats.spearmanr(x, y)[0]


def cosine(v1, v2):
    assert (v1.shape == v2.shape)
    v1  = v1.reshape(1, -1)
    v2 = v2.reshape(1, -1)
    v1 = normalize(v1)
    v2 = normalize(v2)
    return asscalar((v1 * v2).sum(axis=1))
