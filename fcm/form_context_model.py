from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
import jsonpickle

import my_log
from batch_builder import InputProcessor, EndOfDatasetException

# logging options
np.set_printoptions(suppress=True)
tf.logging.set_verbosity(tf.logging.DEBUG)
logger = my_log.get_logger('root')

# constants definition
FORM_ONLY = 'form_only'
CONTEXT_ONLY = 'context_only'
SINGLE_PARAMETER = 'single_parameter'
GATED = 'gated'
DEFAULT = 'default'
CLUSTERING = 'clustering'
COMBINATORS = [FORM_ONLY, CONTEXT_ONLY, SINGLE_PARAMETER, GATED]
CONTEXT_WEIGHTS = [DEFAULT, CLUSTERING]


class RareWordVectorizer(ABC):
    @abstractmethod
    def train(self, num_epochs: int, batch_size: int,
              log_after_n_steps: int = 100, out_path: str = None) -> None:
        pass

    @abstractmethod
    def infer_vector(self, word: str, contexts: List[str]) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'RareWordVectorizer':
        pass


class FormContextModel(RareWordVectorizer):
    def __init__(self, batch_builder: InputProcessor, emb_dim: int = None, learning_rate: float = None,
                 combinator: str = None, sent_weights: str = None, distance_embedding: bool = None):

        self.emb_dim = emb_dim
        self.learning_rate = learning_rate
        self.combinator = combinator
        self.sent_weights = sent_weights
        self.distance_embedding = distance_embedding
        self.batch_builder = batch_builder
        self._setup()

    def _setup(self):
        tf.reset_default_graph()
        tf.set_random_seed(1234)

        self.features = {
            # shape: batch_size x max_ngrams_per_word
            'ngrams': tf.placeholder(tf.int32, shape=[None, None]),
            # shape: batch_size
            'ngram_lengths': tf.placeholder(tf.int32, shape=[None]),
            # shape: batch_size x max_contexts_per_word x max_words_per_context x emb_dim
            'context_vectors': tf.placeholder(tf.float32, shape=[None, None, None, self.emb_dim]),
            # shape: batch_size
            'context_lengths': tf.placeholder(tf.int32, shape=[None]),
            # shape: batch_size x max_contexts_per_word
            'words_per_context': tf.placeholder(tf.int32, shape=[None, None]),
            # shape: batch_size x max_contexts_per_word x max_words_per_context
            'distances': tf.placeholder(tf.int32, shape=[None, None, None])
        }

        self.form_embedding = self._form_embedding(self.features)
        self.context_embedding = self._context_embedding(self.features, self.form_embedding)

        self.form_context_embedding = self._combine_embeddings(
            self.form_embedding, self.context_embedding, self.features)

        self.targets = tf.placeholder(tf.float32, shape=[None, self.emb_dim])

        print(self.targets)
        print(self.form_context_embedding)
        print(self.emb_dim)
        print(self.learning_rate)

        self.loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.form_context_embedding)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

        self.session = tf.Session()

        with self.session.as_default():
            tf.global_variables_initializer().run()
            tf.tables_initializer().run()

    def infer_vector(self, word: str, context: List[str]) -> np.ndarray:

        batch_inputs, _ = self.batch_builder.generate_batch_from_input(word, context)
        feed_dict = {self.features[feature]: batch_inputs[feature] for feature in self.features}

        if batch_inputs['context_vectors'].size == 0:
            if self.combinator == CONTEXT_ONLY:
                logger.warning('Cannot infer embeddings without contexts, returning zeros instead')
                return np.zeros(self.emb_dim)
            vector = self.session.run(self.form_embedding, feed_dict=feed_dict)
        else:
            vector = self.session.run(self.form_context_embedding, feed_dict=feed_dict)

        vector = np.reshape(vector, [self.emb_dim])
        return vector

    def train(self, num_epochs: int, batch_size: int, log_after_n_steps: int = 100,
              out_path: str = None):

        summed_loss, step = 0, 0

        for epoch in range(num_epochs):
            while True:
                try:
                    batch_inputs, batch_labels = self.batch_builder.generate_batch_from_buffer(batch_size)

                    feed_dict = {self.targets: batch_labels}
                    for feature in self.features:
                        feed_dict[self.features[feature]] = batch_inputs[feature]

                    _, loss_val = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)

                    summed_loss += loss_val
                    step += 1

                    if step > 0 and step % log_after_n_steps == 0:
                        logger.info('Step: %d\tLoss: %.17f', step, (summed_loss / log_after_n_steps))
                        summed_loss = 0

                except EndOfDatasetException:
                    logger.info('Done with epoch %d', epoch)

                    if out_path is not None:
                        self.save('{}.e{}'.format(out_path, epoch))

                    self.batch_builder.reset()
                    break

    def _form_embedding(self, features):

        if self.combinator == CONTEXT_ONLY:
            return tf.zeros([1])

        ngrams = features['ngrams']
        word_lengths = features['ngram_lengths']

        with tf.device("/cpu:0"):
            ngram_embeddings = tf.get_variable("ngram_embeddings",
                                               [self.batch_builder.ngram_builder.get_number_of_ngrams(),
                                                self.batch_builder.vector_size])
            ngrams_embedded = tf.nn.embedding_lookup(ngram_embeddings, ngrams, name="ngrams_embedded")

        mask = tf.sequence_mask(word_lengths, dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)

        attention_filtered_sums = tf.reduce_sum(mask, axis=1, keep_dims=True)
        attention = mask / attention_filtered_sums

        return tf.reduce_sum(ngrams_embedded * attention, axis=1)

    def _context_embedding_uniform(self, context_vectors, words_per_context):

        mask = tf.sequence_mask(words_per_context, dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)

        words_per_training_instance = tf.reduce_sum(tf.cast(words_per_context, dtype=tf.float32), axis=1,
                                                    keep_dims=True)
        words_per_training_instance = tf.expand_dims(tf.expand_dims(words_per_training_instance, -1), -1)

        attention = mask / tf.maximum(words_per_training_instance, 1)
        return tf.reduce_sum(context_vectors * attention, axis=[1, 2])

    def _context_embedding(self, features, form_embedding=None):

        if self.combinator == FORM_ONLY:
            return tf.zeros([1])

        elif self.sent_weights == DEFAULT:
            return self._context_embedding_uniform(features['context_vectors'], features['words_per_context'])

        mask = tf.sequence_mask(features['words_per_context'], dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)

        # get weights for each context and expand them to 4 dimensions
        context_weights = self._context_weights(features, form_embedding)
        context_weights = tf.expand_dims(tf.expand_dims(context_weights, -1), -1)

        # get weights for each word and expand them to 4 dimensions
        word_weights = self._word_weights(features)
        word_weights = tf.expand_dims(word_weights, -1)

        weights = mask * (context_weights * word_weights)

        if self.distance_embedding:
            distance_embeddings = tf.get_variable("distance_embeddings_01", [20 + 1, self.batch_builder.vector_size])
            distances_embedded = tf.nn.embedding_lookup(distance_embeddings, features['distances'],
                                                        name="distances_embedded_01")

            vectors_with_distances = features['context_vectors'] * distances_embedded
            context_embedding = tf.reduce_sum(vectors_with_distances * weights, axis=[1, 2])
        else:
            context_embedding = tf.reduce_sum(features['context_vectors'] * weights, axis=[1, 2])
        return context_embedding

    def _context_weights(self, features, form_embedding=None):
        """
        Returns the weight for each context of each traning instance
        The shape is batch_size x max_number_of_contexts
        """
        if self.sent_weights == CLUSTERING:
            # get one embedding per context
            if self.distance_embedding:
                logger.info("using distance embeddings")
                distance_embeddings = tf.get_variable("distance_embeddings_02",
                                                      [20 + 1, self.batch_builder.vector_size])
                distances_embedded = tf.nn.embedding_lookup(distance_embeddings, features['distances'],
                                                            name="distances_embedded_02")
                vectors_with_distances = features['context_vectors'] * distances_embedded
                vectors_avg = tf.reduce_sum(vectors_with_distances, axis=2)

            else:
                vectors_avg = tf.reduce_sum(features['context_vectors'], axis=2)

            vectors_avg /= tf.expand_dims(tf.cast(features['words_per_context'], dtype=tf.float32), axis=-1) + 1e-9

            # compute the pairwise match between the embeddings obtained from all contexts
            vectors_avg = tf.layers.dense(vectors_avg, self.emb_dim, use_bias=None,
                                          activation=None, name='vectors_avg_linear')
            match_scores = FormContextModel.attention_fun(vectors_avg, vectors_avg)

            mask = tf.sequence_mask(features['context_lengths'], dtype=tf.float32)
            mask_key = tf.expand_dims(mask, 1)
            mask_query = tf.expand_dims(mask, 2)

            # batch_size x max_context_length x max_context_length
            match_scores = tf.multiply(match_scores, mask_key)
            match_scores = tf.multiply(match_scores, mask_query)

            self.prx = tf.identity(match_scores)

            # batch_size x max_context_length
            match_scores = tf.reduce_sum(match_scores, axis=-1)

            # modify the match_scores to sum to 1 for each context
            with tf.name_scope("match_scores_softmax"):
                match_scores = tf.multiply(match_scores, mask)
                match_scores_summed = tf.reduce_sum(match_scores, axis=1, keepdims=True)
                match_scores = tf.div(match_scores, match_scores_summed + 1e-9)

            return match_scores

    def _word_weights(self, features):
        """
        Returns the weight for each word within a context, i.e. 1/words_per_context
        The shape is batch_size x max_number_of_contexts x max_number_of_words
        """
        weights = tf.ones(tf.shape(features['context_vectors'])[:3], dtype=tf.float32)
        weights = weights / tf.maximum(1.0,
                                       tf.expand_dims(tf.cast(features['words_per_context'], dtype=tf.float32), -1))
        return weights

    def _combine_embeddings(self, form_embedding, context_embedding, features):

        form_alpha = 0
        context_alpha = 0

        if self.combinator != FORM_ONLY and self.combinator != CONTEXT_ONLY:

            if self.combinator == SINGLE_PARAMETER:
                form_alpha = tf.sigmoid(tf.get_variable("alpha_intra", [1]))
                context_alpha = 1 - form_alpha

            elif self.combinator == GATED:
                combined_guess = tf.concat([form_embedding, context_embedding], axis=-1)
                alpha_kernel = tf.get_variable("alpha_kernel", [2 * self.emb_dim, 1])
                alpha_bias = tf.get_variable("alpha_bias", [1])
                form_alpha = tf.matmul(combined_guess, alpha_kernel) + alpha_bias
                form_alpha = tf.sigmoid(form_alpha)
                context_alpha = 1 - form_alpha

            else:
                raise ValueError("Combinator {} not implemented".format(self.combinator))

            alpha_sum = context_alpha + form_alpha + 1e-9
            form_alpha /= alpha_sum
            context_alpha /= alpha_sum

        if self.combinator != FORM_ONLY:
            context_embedding = tf.layers.dense(context_embedding, self.emb_dim, use_bias=None,
                                                activation=None, name='context_embedding_linear')

        if self.combinator == FORM_ONLY:
            return form_embedding

        elif self.combinator == CONTEXT_ONLY:
            return context_embedding

        else:
            form_alpha = tf.Print(form_alpha, [form_alpha, context_alpha], message='(form, context) = ')
            return form_alpha * form_embedding + context_alpha * context_embedding

    @staticmethod
    def attention_fun(Q, K, scaled_=True):
        attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

        if scaled_:
            d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

        return attention

    def __getstate__(self):

        # TODO solve the other way around (remove unneeded items instead of keeping needed items)
        odict = {
            'emb_dim': self.emb_dim,
            'learning_rate': self.learning_rate,
            'combinator': self.combinator,
            'sent_weights': self.sent_weights,
            'distance_embedding': self.distance_embedding,
            'batch_builder': self.batch_builder
        }
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self._setup()

    def save(self, path: str) -> None:
        logger.info('Saving model to {}'.format(path))
        saver = tf.train.Saver()
        saver.save(self.session, path)
        with open(path + '.config.json', 'w', encoding='utf8') as f:
            f.write(jsonpickle.encode(self))
        logger.info('Done saving model')

    @classmethod
    def load(cls, path: str) -> 'FormContextModel':
        logger.info('Loading model from {}'.format(path))
        with open(path + '.config.json', 'r', encoding='utf8') as f:
            model = jsonpickle.decode(f.read())
            model._setup()
        saver = tf.train.Saver()
        saver.restore(model.session, path)
        logger.info('Done loading model')
        return model
