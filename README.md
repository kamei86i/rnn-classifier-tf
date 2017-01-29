# rnn-classifier-tf
A simple sentence classifier based on Recurrent Neural Networks implemented in Tensorflow.

The program can be used to perform sentence classification. It computes a representation of a sentence with a
recurrent neural network (with a GRU cell) starting from the word embeddings associated to each word of a sentence.
This representation is then classified with a linear classifier.

The data provided comes from the Question Classification dataset, available at http://cogcomp.cs.illinois.edu/Data/QA/QC/,
where no pre-processing is performed (only tokenization on whitespace).
The embeddings are initialised randomly, but one can try to use pre-trained word embeddings.
