import argparse
import io
import my_log
import re

from form_context_model import FormContextModel

logger = my_log.get_logger('root')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)

    args = parser.parse_args()
    model = FormContextModel.load(args.model)

    count = 0

    with io.open(args.input, 'r', encoding='utf-8') as input_file, io.open(args.output, 'w',
                                                                           encoding='utf-8') as output_file:
        for line in input_file.read().splitlines():
            comps = re.split(r'\t', line)
            word = comps[0]
            context = comps[1:]
            context = [c for c in context if c != '']
            vec = model.infer_vector(word, context)
            output_file.write(word + ' ' + ' '.join([str(x) for x in vec]) + '\n')
            count += 1
            if count % 100 == 0:
                logger.info('Done processing %d words', count)


if __name__ == '__main__':
    main()
