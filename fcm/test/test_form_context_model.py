import os
import re
import unittest
import numpy.testing as npt

import my_log
from batch_builder import InputProcessor
from form_context_model import FormContextModel

logger = my_log.get_logger('root')


class TestFormContextModel(unittest.TestCase):
    def setUp(self):
        vocab_file = os.path.join(os.path.dirname(__file__), 'data', 'vocab_with_count.txt')
        embeddings_file = os.path.join(os.path.dirname(__file__), 'data', 'embeddings.txt')

        config = {
            'learning_rate': 0.01,
            'emb_dim': 3,
            'combinator': 'gated',
            'sent_weights': 'clustering'
        }

        batch_builder = InputProcessor(train_files=[], vector_size=3, vocab_file=vocab_file,
                                       word_embeddings_file=embeddings_file, word_embeddings_format='text')
        self.form_context_model = FormContextModel(batch_builder=batch_builder, **config)

    def test_save_and_load(self):

        vec1 = self.form_context_model.infer_vector('word', ['lord word the sword ...'])

        self.form_context_model.save(os.path.join(os.path.dirname('__file'), 'data', 'out', 'fcm.model'))
        form_context_model_loaded = FormContextModel.load(
            os.path.join(os.path.dirname('__file'), 'data', 'out', 'fcm.model'))

        vec2 = form_context_model_loaded.infer_vector('word', ['lord word the sword'])

        npt.assert_array_almost_equal(vec1, vec2)

    def test_infer_vectors(self):
        with open(os.path.join(os.path.dirname(__file__), 'data', 'word_contexts.txt'), 'r',
                  encoding='utf-8') as input_file:
            for line in input_file.read().splitlines():
                comps = re.split(r'\t', line)
                word = comps[0]
                context = comps[1:]
                context = [c for c in context if c != '']
                logger.info('Inferring embedding for "{}" with contexts {}'.format(word, context))
                vec = self.form_context_model.infer_vector(word, context)
