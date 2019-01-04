import argparse
import io
import logging
import re

from form_context_model import load_model

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help="The form-context model file")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="The input file. Each line of this file must be in the format 'WORD<t>CONTEXT1<t>CONTEXT2<t>...<t>CONTEXTn' where <t> denotes a TAB")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="The output file. Each line of this file will be in the format 'WORD VECTOR'")

    args = parser.parse_args()
    model = load_model(args.model)

    count = 0

    with io.open(args.input, 'r', encoding='utf-8') as input_file, io.open(args.output, 'w',
                                                                           encoding='utf-8') as output_file:
        for line in input_file.read().splitlines():
            comps = re.split(r'\t', line)
            word = comps[0]
            context = comps[1:]
            vec = model.infer_vector(word, context)
            output_file.write(word + ' ' + ' '.join([str(x) for x in vec]) + '\n')
            count += 1
            if count % 100 == 0:
                logger.info('Done processing %d words', count)


if __name__ == '__main__':
    import logging.config

    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    main()
