import argparse
import io
import os
import datetime
import socket
import form_context_model as fcm

from dn_evaluator import DefinitionalNonceEvaluator
from crw_evaluator import ContextualizedRareWordsEvaluator
from chimera_evaluator import ChimeraEvaluator
from batch_builder import OOVBatchBuilder

from embedding_loader import get_w2v_model_gensim, get_w2v_model_file

EVALUATORS = ["dn", "crw", "crw_dev", "none", "chimeras"]


def main():

    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--model', '-m', type=str, required=True, help="Where to save the trained model")
    parser.add_argument('--train_dir', type=str, required=True, help="The output directory of the preprocessing script")
    parser.add_argument('--emb_file', type=str, required=True, help="The file where the original word embeddings are stored")
    parser.add_argument('--vocab', type=str, required=True, help="The vocab to use. Each line of this file must be in the format '<word> <count>' where <count> is the number of occurrences of <word> in the training corpus.")
    parser.add_argument('--emb_dim', type=int, required=True, help="The dimensionality of the embeddings")
    parser.add_argument('--emb_format', type=str, choices=['text', 'gensim'], default='text', help="The format in which the original embeddings are stored. Use 'text' if the embeddings are stored in plain text format (i.e. each line is of the form '<word> <vector>') and 'gensim' if the embeddings are stored in gensim's format.")

    # general model parameters
    parser.add_argument('--num_buckets', type=int, default=25, help="The number of buckets used by the preprocessing script")
    parser.add_argument('--batch_size', '-bs', type=int, default=64, help="The batch size to use during training")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help="The initial learning rate")
    parser.add_argument('--num_epochs', '-e', type=int, default=10, help="The number of epochs to train for")

    parser.add_argument('--combinator', choices=fcm.COMBINATORS, default=fcm.GATED, help="The combination function to combine form and context embeddings. See the paper for an explanation of each combination function")
    parser.add_argument('--sent_weights', choices=fcm.CONTEXT_WEIGHTS, default='orig', help="A weighing function for contexts. Simply leave this at its default value to obtain the model described in the paper.")
    parser.add_argument('--min_word_count', '-mwc', type=int, default=100, help="The minimum number of occurrences of a word for it to be used as a training instance")

    # context model parameters
    parser.add_argument('--smin', type=int, default=20, help="The minimum number of contexts sampled for each training instance")
    parser.add_argument('--smax', type=int, default=20, help="The maximum number of contexts sampled for each training instance")

    # word model parameters
    parser.add_argument('--nmin', type=int, default=3, help="The minimum number of characters in an ngram")
    parser.add_argument('--nmax', type=int, default=5, help="The maximum number of characters in an ngram")

    # evaluation parameters
    parser.add_argument('--evaluator', choices=EVALUATORS, default=None, help="The (optional) evaluation dataset to be used after each epoch.")

    # dn evaluation
    parser.add_argument('--dn_eval', type=str,
                        default='../data/oov/n2v_shared_dataset/n2v.definitional.dataset.test.txt', help="The path to the Nonce2Vec test file. This is only required if 'dn' is chosen as an evaluator.")

    # crw evaluation
    parser.add_argument('--rare_vocab', type=str, default='../data/crw/dev/rarevocab.txt')

    # chimera evaluation
    parser.add_argument('--chimeras', type=str, help="The path to the chimeras file. This is only required if 'chimeras' is chosen as an evaluator.")

    parser.add_argument('--blacklist', type=str, default=None, help="A file containing a list of words to be left out during training. Each line should contain exactly one word.")

    parser.add_argument('--log_file', type=str, default=None, help="The log file to which evaluation results are written after each training epoch.")

    args = parser.parse_args()

    if args.emb_format == 'text':
        word2vec = get_w2v_model_file(filename=args.emb_file, dim=args.emb_dim)
    else:
        word2vec = get_w2v_model_gensim(filename=args.emb_file)

    train_files = [args.train_dir + 'train.bucket' + str(i) + '.txt' for i in range(args.num_buckets)]

    if args.combinator == fcm.FORM_ONLY:
        args.sample_context_words = 1

    if args.combinator == fcm.CONTEXT_ONLY:
        args.nmin = 1
        args.nmax = 1

    if args.log_file is not None:
        log_file = io.open(args.log_file, 'a', 1)

    else:
        model_basename = os.path.basename(args.model)
        log_file = io.open('log/' + model_basename + '.log', 'a', 1)

    now = datetime.datetime.now()

    train_server = str(os.environ.get('STY', 'main')) + '@' + str(socket.gethostname())
    log_file.write('-- Training on ' + train_server + ' at ' + now.isoformat() + '  with args = ' + str(args) + ' --\n')

    word_blacklist_file = args.dn_eval if args.evaluator == "dn" else args.rare_vocab

    if args.evaluator == "chimeras" or args.evaluator is None:
        word_blacklist_file = None

    if args.blacklist:
        word_blacklist_file = [word_blacklist_file, args.blacklist]
        logging.info("Word blacklist files: " + str(word_blacklist_file))

    batch_builder = OOVBatchBuilder(
        word2vec=word2vec,
        test_words_file=word_blacklist_file,
        data_files=train_files,
        unigram_file=args.vocab,
        sample_sentences=[args.smin, args.smax],
        ngram_range=[args.nmin, args.nmax],
        emb_dim=args.emb_dim,
        ngram_dropout=0,
        min_word_count=args.min_word_count)

    config = {
        'learning_rate': args.learning_rate,
        'emb_dim': args.emb_dim,
        'batch_size': args.batch_size,
        'combinator': args.combinator,
        'sent_weights': args.sent_weights,
        'ngram_vocab_size': batch_builder.nr_of_words,
        'nmin': args.nmin,
        'nmax': args.nmax,
        'vocab_file': args.vocab,
        'word_blacklist_file': word_blacklist_file,
        'base_embeddings': {
            'format': args.emb_format,
            'file': args.emb_file
        },
    }

    evaluator = None

    if args.evaluator == "dn":
        evaluator = DefinitionalNonceEvaluator(word2vec=word2vec, eval_file=args.dn_eval, log_file=log_file)
    elif args.evaluator == "chimeras":
        evaluator = ChimeraEvaluator(log_file=log_file, word2vec=word2vec, chimeras_dir=args.chimeras, train=True)
    elif args.evaluator == "crw" or args.evaluator == "crw_dev":
        evaluator = ContextualizedRareWordsEvaluator(log_file=log_file,
                                                     dataset='test' if args.evaluator == "crw" else 'dev')

    model = fcm.FormContextModel(config, batch_builder)
    model.train(num_epochs=args.num_epochs, evaluator=evaluator, model_path=args.model)

    log_file.write('\n')
    log_file.close()

    logging.info('training complete')


if __name__ == '__main__':
    import logging.config

    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    main()
