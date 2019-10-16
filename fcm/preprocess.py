import random
from typing import List
from collections import Counter
import nltk
import os
import argparse

import my_log

FILE_NAME = 'train'
SHUFFLED_SUFFIX = '.shuffled'
TOKENIZED_SUFFIX = '.tokenized'
VOCAB_SUFFIX = '.voc'
VOCAB_WITH_COUNTS_SUFFIX = '.vwc'
BUCKET_SUFFIX = '.bucket'

logger = my_log.get_logger('root')


def preprocess_training_file(in_path: str, out_dir: str, shuffle=True, tokenize=True, seed: int = 1234,
                             min_count_target: int = 100, min_count_context: int = 0, num_buckets: int = 25,
                             max_contexts_per_word=1000, max_context_size=25,
                             remove_one_word_lines: bool = True) -> None:
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        if os.listdir(out_dir):
            logger.warning("Directory {} is not empty".format(out_dir))
    else:
        raise FileNotFoundError("Directory {} doesn't exists".format(out_dir))

    file_root = os.path.join(out_dir, FILE_NAME)

    shuffled_path = file_root + SHUFFLED_SUFFIX if shuffle else in_path
    tokenized_path = file_root + (SHUFFLED_SUFFIX if shuffle else '') + TOKENIZED_SUFFIX if tokenize else shuffled_path
    bucket_path = file_root + BUCKET_SUFFIX

    # step 1: shuffle the file
    if shuffle:
        _shuffle_lines(in_path, shuffled_path, seed, remove_one_word_lines=remove_one_word_lines)

    # step 2: tokenize and lowercase the file
    if tokenize:
        _tokenize_lines(shuffled_path, tokenized_path)

    # step 3: create the vocab files required for various succeeding steps
    _create_vocab(tokenized_path, file_root + VOCAB_SUFFIX + str(min_count_context),
                  with_counts=False, min_count=min_count_context)
    _create_vocab(tokenized_path, file_root + VOCAB_SUFFIX + str(min_count_target),
                  with_counts=False, min_count=min_count_target)
    _create_vocab(tokenized_path, file_root + VOCAB_WITH_COUNTS_SUFFIX + str(min_count_context),
                  with_counts=True, min_count=min_count_context)
    _create_vocab(tokenized_path, file_root + VOCAB_WITH_COUNTS_SUFFIX + str(min_count_target),
                  with_counts=True, min_count=min_count_target)

    vocab_targets = _load_vocab(file_root + VOCAB_SUFFIX + str(min_count_target), with_counts=False)

    # to make all buckets approximately equal in size, we shuffle the vocab before distributing it
    _shuffle_vocab(vocab_targets, seed=seed)

    # step 4: split the target vocab into buckets
    words_per_bucket = int(len(vocab_targets) / num_buckets) + 1
    logger.info(
        'Distributing {} words into {} buckets with {} words'.format(len(vocab_targets), num_buckets, words_per_bucket))

    for i in range(num_buckets):
        bucket_targets = vocab_targets[words_per_bucket * i: words_per_bucket * (i + 1)]
        if not bucket_targets:
            logger.warning('No words (of {}) left for bucket {} of {}'.format(len(vocab_targets), i + 1, num_buckets))
            continue
        logger.info('Creating bucket {} of {} with {} words'.format(i + 1, num_buckets, len(bucket_targets)))
        create_context_file(tokenized_path, bucket_path + str(i) + '.txt', bucket_targets,
                            max_contexts_per_word=max_contexts_per_word,
                            max_context_size=max_context_size)


def _shuffle_vocab(vocab: List[str], seed: int) -> None:
    random.seed(seed)
    random.shuffle(vocab)


def _shuffle_lines(in_path: str, out_path: str, seed: int, remove_one_word_lines: bool) -> None:
    random.seed(seed)

    logger.info('Loading file {} into memory'.format(in_path))

    with open(in_path, 'r', encoding='utf8') as f:
        lines = f.read().splitlines()

    if remove_one_word_lines:
        old_len = len(lines)
        lines = [x for x in lines if len(x.split()) > 1]
        logger.info('Removed {} lines that contained only one word'.format(old_len - len(lines)))

    logger.info('Shuffling all {} lines'.format(len(lines)))

    random.shuffle(lines)

    logger.info('Saving shuffled lines to file {}'.format(out_path))
    with open(out_path, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(line + '\n')


def _tokenize_lines(in_path: str, out_path: str) -> None:
    logger.info('Tokenizing sentences from file {} to {}'.format(in_path, out_path))

    with open(in_path, 'r', encoding='utf8') as f_in, open(out_path, 'w', encoding='utf8') as f_out:
        line_count = 0
        for line in f_in:
            f_out.write(_tokenize_line(line) + '\n')
            line_count += 1
            if line_count % 10000 == 0:
                logger.info('Done tokenizing {} lines'.format(line_count))

    logger.info('Done with tokenization of {} lines'.format(line_count))


def _tokenize_line(line: str) -> str:
    tokens = nltk.word_tokenize(line)
    return ' '.join(token.lower() for token in tokens)


def _create_vocab(in_path: str, out_path: str, with_counts: bool = False, min_count: int = -1):
    logger.info('Creating vocab from file {} (with_counts={}, min_count={})'.format(in_path, with_counts, min_count))

    vocab = Counter()
    with open(in_path, 'r', encoding='utf8') as f:
        for line in f:
            words = line.split()
            vocab.update(words)

    with open(out_path, 'w', encoding='utf8') as f:
        for (word, count) in vocab.most_common():
            if count >= min_count:
                content = word if not with_counts else word + ' ' + str(count)
                f.write(content + '\n')

    logger.info('Done writing vocab to {}'.format(out_path))


def _load_vocab(in_path: str, with_counts: bool = False):
    if with_counts:
        vocab = {}
    else:
        vocab = []

    with open(in_path, 'r', encoding='utf8') as f:
        for line in f.read().splitlines():
            if with_counts:
                word, count = line.split()
                vocab[word] = count
            else:
                vocab.append(line)

    return vocab


def create_context_file(in_path: str, out_path: str, vocab: List[str], max_contexts_per_word=1000,
                        max_context_size=25, tokenize=False):
    vocab = set(vocab)
    contexts_per_word = {}

    for word in vocab:
        contexts_per_word[word] = set()

    with open(in_path, 'r', encoding='utf8') as f:
        line_count = 0
        for composite_line in f:
            composite_line = composite_line[:-1]
            lines = composite_line.split(' .')
            for line in lines:
                if tokenize:
                    line = _tokenize_line(line)
                words = line.split()
                for idx, word in enumerate(words):
                    if word in vocab and len(contexts_per_word[word]) < max_contexts_per_word:
                        min_idx = max(0, idx - max_context_size)
                        max_idx = idx + max_context_size
                        contexts_per_word[word].add(' '.join(words[min_idx:max_idx]))
            line_count += 1
            if line_count % 100000 == 0:
                logger.info('Done processing {} lines'.format(line_count))

    with open(out_path, 'w', encoding='utf8') as f:
        for word in contexts_per_word.keys():
            contexts = list(contexts_per_word[word])
            f.write(word + '\t' + '\t'.join(contexts) + '\n')
    logger.info('Done writing bucket to {}'.format(out_path))


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('mode', choices=['train', 'test'],
                        help="Whether to preprocess a file for training or testing (i.e. inferring embeddings for novel words)")
    parser.add_argument('--input', type=str, required=True,
                        help="The path of the raw text file from which (word, context) pairs are to be extracted")
    parser.add_argument('--output', type=str, required=True,
                        help="For training: the output directory in which all files required for training are stored. "
                             "For testing: the single file in which (word, context) pairs are stored.")

    # training + testing arguments
    parser.add_argument('--max_contexts_per_word', type=int, default=1000,
                        help="The maximum number of contexts to be stored per word")
    parser.add_argument('--max_context_size', type=int, default=25,
                        help="The maximum number of context words to the left and right of a word to be considered")

    # training arguments
    parser.add_argument('--no_shuffle', action='store_true',
                        help="If set to true, the training dataset is not shuffled.")
    parser.add_argument('--no_tokenize', action='store_true',
                        help="If set to true, the training dataset is not tokenized.")
    parser.add_argument('--seed', type=int, default=1234,
                        help="The seed used for shuffling the training dataset.")
    parser.add_argument('--min_count_target', type=int, default=100,
                        help="The minimum number of occurrences in the --input file for a word to be included in the output")
    parser.add_argument('--min_count_context', type=int, default=0,
                        help="The minimum number of occurrences in the --input file for a word to be used as a context word")
    parser.add_argument('--num_buckets', type=int, default=25,
                        help="The number of training buckets (or chunks) among which the training data is divided")
    parser.add_argument('--keep_one_word_lines', action='store_true',
                        help="If set to true, lines containing only a single word (before tokenization) are kept in the training dataset.")

    # testing arguments
    parser.add_argument('--words', type=str,
                        help="The path to a file containing the words for which (word, context) pairs are to be created. "
                             "Each line must contain exactly one word. Only required if mode == test.")

    args = parser.parse_args()

    if args.mode == 'train':
        preprocess_training_file(args.input, args.output, shuffle=not args.no_shuffle, tokenize=not args.no_tokenize,
                                 seed=args.seed,
                                 min_count_target=args.min_count_target, min_count_context=args.min_count_context,
                                 num_buckets=args.num_buckets,
                                 max_contexts_per_word=args.max_contexts_per_word,
                                 max_context_size=args.max_context_size,
                                 remove_one_word_lines=not args.keep_one_word_lines)

    else:
        if not args.words:
            raise ValueError("--words must be specified when mode == test")

        words = _load_vocab(args.words, with_counts=False)
        create_context_file(args.input, args.output, words,
                            max_contexts_per_word=args.max_contexts_per_word,
                            max_context_size=args.max_context_size, tokenize=not args.no_tokenize)

        # preprocess_training_file(r'/nfs/datm/schickt/bert-experiments/fcm/train-2/WestburyLab.Wikipedia.Corpus.txt',
        #                         r'/nfs/datm/schickt/bert-experiments/fcm/train-2/', shuffle=True, tokenize=True)

        # create_context_file(r'/nfs/datm/schickt/bert-experiments/fcm/train-2/train.shuffled.tokenized',
        #                    r'/nfs/datm/schickt/bert-experiments/fcm/contexts/base-words-contexts-new-preprocessor-2.txt',
        #                    _load_vocab(r'/nfs/datm/schickt/bert-experiments/vocab-eval-dataset-base-words.txt',
        #                                with_counts=False)
        #                    )


if __name__ == '__main__':
    # _shuffle_lines('test/data/tokenize_test.txt', 'test/data/out/tokenize_test.txt', 1234, True)
    main()
