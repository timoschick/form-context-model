import argparse
import io
import os
import datetime
import socket
import form_context_model as fcm
import my_log

from batch_builder import InputProcessor

logger = my_log.get_logger('root')


def main():
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--model', '-m', type=str, required=True,
                        help="The name of the model")
    parser.add_argument('--train_dir', type=str, required=True,
                        help="The directory in which the buckets for training are stored")
    parser.add_argument('--vocab', type=str, required=True,
                        help="The file in which the vocabulary to be used for training is stored."
                             "Each line should be of the form <WORD> <COUNT>")
    parser.add_argument('--emb_file', type=str, required=True,
                        help="The file in which the embeddings for mimicking and context embeddings"
                             "are stored")
    parser.add_argument('--emb_dim', type=int, required=True,
                        help="The number of dimensions for the given embeddings")

    # general model parameters
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', '-e', type=int, default=10)
    parser.add_argument('--combinator', choices=fcm.COMBINATORS, default=fcm.GATED)
    parser.add_argument('--sent_weights', choices=fcm.CONTEXT_WEIGHTS, default=fcm.CLUSTERING)
    parser.add_argument('--min_word_count', '-mwc', type=int, default=100)
    parser.add_argument('--remove_punctuation', action='store_true')

    # context model parameters
    parser.add_argument('--smin', type=int, default=20)
    parser.add_argument('--smax', type=int, default=20)
    parser.add_argument('--distance_embedding', '-de', action='store_true')

    # word model parameters
    parser.add_argument('--nmin', type=int, default=3)
    parser.add_argument('--nmax', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0)

    # training parameters
    parser.add_argument('--num_buckets', type=int, default=25)
    parser.add_argument('--emb_format', type=str, choices=['text', 'gensim'], default='text')
    parser.add_argument('--log_file', type=str, default=None)

    args = parser.parse_args()

    train_files = [args.train_dir + 'train.bucket' + str(i) + '.txt' for i in range(args.num_buckets)]

    if args.combinator == fcm.FORM_ONLY:
        args.sample_context_words = 1

    elif args.combinator == fcm.CONTEXT_ONLY:
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

    batch_builder = InputProcessor(
        word_embeddings_file=args.emb_file,
        word_embeddings_format=args.emb_format,
        train_files=train_files,
        vocab_file=args.vocab,
        vector_size=args.emb_dim,
        nmin=args.nmin,
        nmax=args.nmax,
        ngram_dropout=args.dropout,
        smin=args.smin,
        smax=args.smax,
        min_word_count=args.min_word_count,
        keep_punctuation=not args.remove_punctuation
    )

    model = fcm.FormContextModel(
        batch_builder=batch_builder,
        emb_dim=args.emb_dim,
        combinator=args.combinator,
        sent_weights=args.sent_weights,
        distance_embedding=args.distance_embedding,
        learning_rate=args.learning_rate
    )

    model.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        out_path=args.model
    )

    log_file.write('\n')
    log_file.close()

    logger.info('Training is complete')


if __name__ == '__main__':
    main()
