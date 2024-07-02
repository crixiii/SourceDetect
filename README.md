# SourceDetect
This pipeline utilises Keras and Tensorflow to train and apply a convolutional neural network model to perform a transient event search through TESS data.

To install this package run `pip install git+https://github.com/andrewmoore73/SourceDetect.git` in terminal. This will download the core package alongside default training/test datasets and a default CNN model. The datasets can be used to train your own custom model or the model, which has already been fully trained and tested on these datasets, can be called directly to skip the training stage entirely.

Check out `example.ipynb` for a brief walkthrough on how to use `SourceDetect`.
