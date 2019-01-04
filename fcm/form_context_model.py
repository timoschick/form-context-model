import numpy as np
import tensorflow as tf
import json
import os
import io
import logging

from batch_builder import load_builder, EndOfDatasetException

logger = logging.getLogger(__name__)

np.set_printoptions(suppress=True)
tf.logging.set_verbosity(tf.logging.DEBUG)

FORM_ONLY = 'form_only'
CONTEXT_ONLY = 'context_only'
GATED = 'gated'
SINGLE_PARAMETER = 'single_parameter'

COMBINATORS = [FORM_ONLY, CONTEXT_ONLY, GATED, SINGLE_PARAMETER]

DEFAULT = 'orig'
POSSUM = 'possuml'
CLUSTERING = 'clustering'

CONTEXT_WEIGHTS = [DEFAULT, POSSUM, CLUSTERING]


class FormContextModel:
    def __init__(self, config, batch_builder=None):

        tf.reset_default_graph()
        tf.set_random_seed(1234)

        self.config = config
        self.batch_builder = batch_builder

        if self.batch_builder is None:
            self.batch_builder = load_builder(config)

        batch_size = None

        self.features = {
            # word features
            'ngrams': tf.placeholder(tf.int32, shape=[batch_size, None]),
            'ngram_lengths': tf.placeholder(tf.int32, shape=[batch_size]),
            # context features
            'context_vectors': tf.placeholder(tf.float32, shape=[batch_size, None, None, self.config['emb_dim']]),
            'context_lengths': tf.placeholder(tf.int32, shape=[batch_size]),
            'words_per_context': tf.placeholder(tf.int32, shape=[batch_size, None]),
            'distances': tf.placeholder(tf.int32, shape=[batch_size, None, None])
        }

        self.targets = tf.placeholder(tf.float32, shape=[batch_size, self.config['emb_dim']])

        self.form_embedding = self._form_embedding(self.features)
        self.context_embedding = self._context_embedding(self.features)

        self.form_context_embedding = self._combine_embeddings(
            self.form_embedding, self.context_embedding, self.features)

        self.loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.form_context_embedding)

        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
        self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

        self.session = tf.Session()

        with self.session.as_default():
            tf.global_variables_initializer().run()
            tf.tables_initializer().run()

    def train(self, num_epochs=1, evaluator=None, model_path=None, print_after_n_steps=100):

        summed_loss, step = 0, 0

        for epoch in range(num_epochs):
            while True:
                try:
                    batch_inputs, batch_labels = self.batch_builder.generate_batch_from_buffer(
                        self.config['batch_size'])

                    feed_dict = {self.targets: batch_labels}
                    for feature in self.features:
                        feed_dict[self.features[feature]] = batch_inputs[feature]

                    _, loss_val = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)

                    summed_loss += loss_val
                    step += 1

                    if step > 0 and step % print_after_n_steps == 0:
                        logger.info('Step: %d\tLoss: %.17f', step, (summed_loss / print_after_n_steps))
                        summed_loss = 0

                except EndOfDatasetException:
                    logger.info('Done with epoch %d', epoch)

                    if model_path is not None:
                        self.save(model_path + '.e' + str(epoch))

                    if evaluator:
                        logger.info('Starting evaluation')
                        evaluator.evaluate_model(self)
                        logger.info('Finished evaluation')

                    self.batch_builder.reset()
                    break

    def _form_embedding(self, features):

        ngrams = features['ngrams']
        word_lengths = features['ngram_lengths']

        if self.config['combinator'] == CONTEXT_ONLY:
            return tf.zeros([1])

        with tf.device("/cpu:0"):
            ngram_embeddings = tf.get_variable("ngram_embeddings",
                                               [self.batch_builder.nr_of_letters, self.batch_builder.emb_dim])
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

    def _context_embedding(self, features):

        if self.config['combinator'] == FORM_ONLY:
            return tf.zeros([1])

        elif self.config['sent_weights'] == 'orig':
            return self._context_embedding_uniform(features['context_vectors'], features['words_per_context'])

        mask = tf.sequence_mask(features['words_per_context'], dtype=tf.float32)
        mask = tf.expand_dims(mask, -1)

        # get weights for each context and expand them to 4 dimensions
        context_weights = self._context_weights(features)
        context_weights = tf.expand_dims(tf.expand_dims(context_weights, -1), -1)

        # get weights for each word and expand them to 4 dimensions
        word_weights = self._word_weights(features)
        word_weights = tf.expand_dims(word_weights, -1)

        weights = mask * (context_weights * word_weights)

        context_embedding = tf.reduce_sum(features['context_vectors'] * weights, axis=[1, 2])
        return context_embedding

    def _context_weights(self, features):
        """
        Returns the weight for each context of each traning instance
        The shape is batch_size x max_number_of_contexts
        """
        if self.config['sent_weights'] == CLUSTERING:
            # get one embedding per context
            vectors_avg = tf.reduce_sum(features['context_vectors'], axis=2)
            vectors_avg /= tf.expand_dims(tf.cast(features['words_per_context'], dtype=tf.float32), axis=-1) + 1e-9

            # compute the pairwise match between the embeddings obtained from all contexts
            vectors_avg = tf.layers.dense(vectors_avg, self.config['emb_dim'], use_bias=None,
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

            # TODO perhaps multiple iterations
            return match_scores

        elif self.config['sent_weights'] == POSSUM:

            distance_embeddings = tf.get_variable("distance_embeddings", [20 + 1, self.batch_builder.emb_dim])
            distances_embedded = tf.nn.embedding_lookup(distance_embeddings, features['distances'],
                                                        name="distances_embedded")

            vectors_with_distances = features['context_vectors'] * distances_embedded
            vectors_with_distances = tf.layers.dense(vectors_with_distances, self.config['emb_dim'], use_bias=True,
                                                     activation=None, name='vectors_with_distances_linear')

            vectors_summed = tf.reduce_sum(vectors_with_distances, axis=2)

            attention_score = tf.layers.dense(vectors_summed, 1, activation=None, name='att_score')
            attention_score = tf.squeeze(attention_score, axis=2)
            context_length_mask = tf.sequence_mask(features['context_lengths'], dtype=tf.float32)

            with tf.name_scope("softmax"):
                attention_score = tf.exp(attention_score)
                attention_score_masked = tf.multiply(attention_score, context_length_mask)
                attention_sum = tf.reduce_sum(attention_score_masked, axis=1, keepdims=True)
                attention = attention_score / (attention_sum + 1e-9)

            return attention

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

        if self.config['combinator'] != FORM_ONLY and self.config['combinator'] != CONTEXT_ONLY:

            if self.config['combinator'] == SINGLE_PARAMETER:
                form_alpha = tf.sigmoid(tf.get_variable("alpha_intra", [1]))
                context_alpha = 1 - form_alpha

            elif self.config['combinator'] == GATED:
                combined_guess = tf.concat([form_embedding, context_embedding], axis=-1)
                alpha_kernel = tf.get_variable("alpha_kernel", [2 * self.config['emb_dim'], 1])
                alpha_bias = tf.get_variable("alpha_bias", [1])
                form_alpha = tf.matmul(combined_guess, alpha_kernel) + alpha_bias
                form_alpha = tf.sigmoid(form_alpha)
                context_alpha = 1 - form_alpha

            else:
                raise ValueError("combinator not implemented")

            alpha_sum = context_alpha + form_alpha + 1e-9
            form_alpha /= alpha_sum
            context_alpha /= alpha_sum

        if self.config['combinator'] != FORM_ONLY:
            context_embedding = tf.layers.dense(context_embedding, self.config['emb_dim'], use_bias=None,
                                                activation=None, name='context_embedding_linear')

        if self.config['combinator'] == FORM_ONLY:
            return form_embedding

        elif self.config['combinator'] == CONTEXT_ONLY:
            return context_embedding

        else:
            return form_alpha * form_embedding + context_alpha * context_embedding

    @staticmethod
    def attention_fun(Q, K, scaled_=True):
        attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

        if scaled_:
            d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

        return attention

    def infer_vector(self, word, context):

        batch_inputs, batch_labels = self.batch_builder.generate_batch_from_context(word, context)
        feed_dict = {self.targets: batch_labels}
        for feature in self.features:
            feed_dict[self.features[feature]] = batch_inputs[feature]

        if batch_inputs['context_vectors'].size == 0:
            if self.config['combinator'] == CONTEXT_ONLY:
                logger.warn(
                    'Cannot infer embedding without contexts when combinator = CONTEXT_ONLY, returning zeros instead')
                return np.zeros(self.config['emb_dim'])
            vector = self.session.run(self.form_embedding, feed_dict=feed_dict)
        else:
            vector = self.session.run(self.form_context_embedding, feed_dict=feed_dict)

        vector = np.reshape(vector, [self.config['emb_dim']])
        return vector

    def save(self, path):
        logger.info('Saving model to %s', path)
        with io.open(path + '.config.json', 'w') as f:
            json.dump(self.config, f)
        saver = tf.train.Saver()
        saver.save(self.session, path)
        logger.info('Done saving model')


def load_model(path):
    logger.info('Loading model from %s', path)
    with open(path + '.config.json') as f:
        config = json.load(f)

    model = FormContextModel(config)

    saver = tf.train.Saver()
    saver.restore(model.session, path)
    logger.info('Done loading model')
    return model
