import argparse
import io

from crw_evaluator import ContextualizedRareWordsEvaluator
from dn_evaluator import DefinitionalNonceEvaluator
from chimera_evaluator import ChimeraEvaluator
from form_context_model import load_model

EVALUATORS = ["dn", "crw", "crw_dev", "chimeras"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='models/tester_model.ckpt')
    parser.add_argument('--evaluator', choices=EVALUATORS, default="dn")

    # dn evaluation
    parser.add_argument('--dn_eval', type=str,
                        default='../data/oov/n2v_shared_dataset/n2v.definitional.dataset.test.txt')

    parser.add_argument('--chimeras', type=str)

    args = parser.parse_args()
    model = load_model(args.model)
    log_file = io.open('log/logxy.txt', 'a', 1)

    if args.evaluator == "dn":
        evaluator = DefinitionalNonceEvaluator(word2vec=model.batch_builder.word2vec, eval_file=args.dn_eval,
                                               log_file=log_file)
    elif args.evaluator == "chimeras":
        evaluator = ChimeraEvaluator(word2vec=model.batch_builder.word2vec, chimeras_dir = args.chimeras)
    else:
        evaluator = ContextualizedRareWordsEvaluator(log_file=log_file,
                                                     dataset='test' if args.evaluator == 'crw' else 'dev')

    print(model.infer_vector("apple", "there is an apple on the tree"))
    evaluator.evaluate_model(model)


if __name__ == '__main__':
    import logging.config

    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    main()
