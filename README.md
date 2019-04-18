# Form-Context Model

This repository contains the code and supplementary material for [Learning Semantic Representations for Novel Words: Leveraging Both Form and Context](https://arxiv.org/abs/1811.03866) and [Attentive mimicking: Better word embeddings by attending to informative contexts](https://arxiv.org/abs/1904.01617).
The contents of each subdirectory are as follows:

**Important**: The code found in this directory is a beautified and easier-to-use version of the original form-context model. Due to random parameter initialization, results may slightly deviate from the ones reported in the papers mentioned above. If you want to *use* the form-context model, this is the right version for you. If, instead, you want to *reprocude* the original results, use the [naacl](https://github.com/timoschick/form-context-model/tree/naacl) branch for the NAACL paper results or contact me via `timo.schick<at>sulzer.de` for the AAAI results.  

## Usage

To train your own instance of the form-context model (FCM) or Attentive Mimicking (AM), you require:

- a large text corpus (e.g., the [Westbury Wikipedia corpus](http://www.psych.ualberta.ca/~westburylab/downloads/westburylab.wikicorp.download.html) used in the papers cited above)
- a set of pretrained word embeddings (e.g. Glove or Word2Vec)

### Preprocessing

Before training the model, you need to preprocess the text corpus. This can be done using the `fcm/preprocess.py` script:

    python3 fcm/preprocess.py train --input PATH_TO_YOUR_TEXT_CORPUS --output TRAINING_DIRECTORY

If you leave all other parameters unchanged, this creates the following files in the specified output directory: 

- `train.shuffled`: a shuffled version of your input corpus;
- `train.shuffled.tokenized`: a shuffled, tokenized and lowercased version of your input corpus;
- `train.vocX`, `train.vwcX`: vocabulary file containing all words that occur at least X times. In the `voc` format, each line contains exactly one word. In the `vwc` format, each line is of the form `word count`;
- `train.bucketX`: a bucket (or chunk) of training instances. Each line is of the form `word<TAB>context1<TAB>context2<TAB>...`

To get an overview of additional parameters for the preprocessing script, run `python3 fcm/preprocess.py -h`.

### Training

To train a new model, use the `fcm/train.py` script:

    python3 fcm/train.py -m MODEL_PATH 
    --train_dir TRAINING_DIRECTORY 
    --emb_file PATH_TO_YOUR_WORD_EMBEDDINGS 
    --emb_dim DIMENSIONALITY_OF_YOUR_WORD_EMBEDDINGS 
    --vocab PATH_TO_THE_TRAIN.VWC100_FILE

By default, the training script uses Attentive Mimicking. If you instead want to train the original FCM, you must pass `--sent_weights default`. Again, an overview of additional parameters for the training script can be obtained via `python3 fcm/train.py -h`.

### Inference

Inferring embeddings for novel words requires a file where each line is of the form `novel_word<TAB>context1<TAB>context2<TAB>...`. If you do not have a such file, you can generate it using the preprocessing script and a `.voc`-file containing all the words you want embeddings for:

    python3 fcm/preprocess.py test --input PATH_TO_YOUR_TEXT_CORPUS --output PATH_TO_THE_TEST_FILE --words PATH_TO_A_VOC_FILE

The acutal inference can then be done using the `fcm/infer_vectors.py` script:

    python3 fcm/infer_vectors.py -m MODEL_PATH -i PATH_TO_THE_TEST_FILE -o PATH_TO_THE_OUTPUT_FILE
    
The specified output file will then contain lines of the form `word embedding`.

## Resources

### crw-dev

This directory contains the CRW development dataset. For more info, refer to the AAAI paper.

### vecmap

This directory contains the VecMap dataset. For more info, refer to the NAACL paper.

## Citation

If you make use of the VecMap dataset or Attentive Mimicking, please cite the following paper:
```
@inproceedings{schick2019attentive,
  title={Attentive mimicking: Better word embeddings by attending to informative contexts},
  author={Schick, Timo and Sch{\"u}tze, Hinrich},
  url="https://arxiv.org/abs/1904.01617",
  booktitle={Proceedings of the Seventeenth Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2019}
}
```
If you make use of the CRW development set or the original form-context model, please cite the following paper:
```
@inproceedings{schick2019learning,
  title={Learning Semantic Representations for Novel Words: Leveraging Both Form and Context},
  author={Schick, Timo and Sch{\"u}tze, Hinrich},
  url="https://arxiv.org/abs/1811.03866",
  booktitle={Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence},
  year={2019}
}
```
