# Form-Context Model

This directory contains the code required for training and using the form-context model. Note that before training, the training corpus must be preprocessed (see `../preprocessing`)

To train a new form-context model, use `train.py`. To infer embeddings for novel words using a trained form-context model, use `infer_vectors.py`.
For both scripts, a commented list of all arguments is available via the `--help` flag.