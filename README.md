# rnn-classifier-tf

A simple sentence classifier based on Recurrent Neural Networks implemented in Tensorflow.

The program can be used to perform sentence classification. It computes a representation of a sentence with a
recurrent (GRU cell) neural network starting from the word embeddings associated to each word of a sentence.
This representation is then classified with a linear classifier.

The data provided comes from the well-known Question Classification dataset, available at http://cogcomp.cs.illinois.edu/Data/QA/QC/,
where no pre-processing is performed (only tokenization on whitespace).
The embeddings are initialised with random vectors. Obviously, it can be possible to use pre-trained word embeddings (
for example word2vec embeddings). It is expected that using pre-trained word embeddings performances will be higher.

## Dependencies
- Python 3
- Tensorflow 0.12.1

## How to train the network
The network can be trained with the *train.py* script; default parameters are already set in the file.

    python train.py
will start training with 2 gru cells, each of with 64 units, with an embedding size of 128, a dropout keep probability set to 0.8
and lambda regularization parameter of 1.
The training will be performed with batches of 128 example over 15 epochs.

You can override these settings by passing them as arguments, for example:

    python train.py --rnn_num 4 --batch_size 256
will override the number of rnn cells and the batch size.
For a complete list of parameters, refer to the beginning of the train.py file. 

The training will save a tensorflow model in the directory runs_qc, while tensorboard summaries are saved in the directory summary_qc.

