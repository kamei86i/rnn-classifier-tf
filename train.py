import os
import random

import numpy as np
import tensorflow as tf
from embedding import *

from rnn_text_classifier import RnnTextClassifier
from sentence import Sentence

# Data parameters
tf.flags.DEFINE_string("data_dir", "data_qc",
                       "The path of the tweet training data_qc")
tf.flags.DEFINE_string("model_dir", "runs_qc",
                       "The path where to save the models")
tf.flags.DEFINE_string("summaries_dir", "summary_qc",
                       "The path where to save the summaries")

# Model parameters
tf.flags.DEFINE_integer("rnn_num", 2, "The numbers of rnn cells (default: 2)")
tf.flags.DEFINE_integer("cell_size", 64, "The size of the rnn cell (default: 64)")
tf.flags.DEFINE_integer("embedding_size", 128,
                        "The size of the embeddings; this value is ignored if a pre-trained embedding is used. (default: 128)")
tf.flags.DEFINE_boolean("train_embedding", False,
                        "Train or not the embeddings when using pre-trained ones (default: False)")
tf.flags.DEFINE_string("embedding_path",
                       "",
                       "The path of an embedding in word2vec binary format")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("lam", 1, "Regularization parameter (default: 1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 15)")
tf.flags.DEFINE_float("learning_rate", 0.001, "set the learning rate for the AdamOptimizer (default: 0.001)")

# Misc Parameters
tf.flags.DEFINE_integer("processors_num", 4, "Number of processors")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

if not os.path.exists(FLAGS.model_dir):
    os.mkdir(FLAGS.model_dir)


def load_file(path):
    f = open(path, "r", encoding='ISO-8859-1')
    dataset = []
    labels = set()
    length = 0
    for line in f:
        sp = line.split()
        label = sp[0].split(":")[0]
        labels.add(label)
        sentence = " ".join(sp[1:])
        tokenized_sentence = sp[1:]
        if len(tokenized_sentence) > length:
            length = len(tokenized_sentence)
        dataset.append(Sentence(sentence, label, tokenized_sentence))

    return dataset, length, labels


def build_vocab(dataset):
    print("Building vocab")
    vocab = dict()
    c = 0
    vocab[PAD_TOKEN] = c
    c += 1
    vocab[UNK_TOKEN] = c
    c += 1

    for d in dataset:
        for token in d.tokens:
            if token not in vocab:
                vocab[token] = c
                c += 1
    print("vocab size is: " + str(len(vocab)))
    return vocab


def build_label_dict(labels):
    ret = dict()
    i = 0
    for label in labels:
        ret[label] = i
        i += 1
    return ret


def load_qc_data(dir):
    print("Loading data...")
    train, tr_max, labels = load_file(dir + "/train_5500.label")
    num_c = len(labels)
    test, tt_max, _ = load_file(dir + "/TREC_10.label")
    max_length = max(tr_max, tt_max)
    print("Max length is: " + str(max_length))
    print("Num classes is: " + str(num_c))
    vocab = build_vocab(train)
    label_dict = build_label_dict(labels)

    for example in train:
        example.pad_to(max_length, PAD_TOKEN)
        example.apply_vocabs(vocab, UNK_TOKEN, label_dict)
    for example in test:
        example.pad_to(max_length, PAD_TOKEN)
        example.apply_vocabs(vocab, UNK_TOKEN, label_dict)
    print("Data loaded")
    return train, test, vocab, max_length, labels


def shuffle(x):
    x = list(x)
    random.shuffle(x)
    return x


def split(data, factor):
    shuffled = shuffle(data)
    train_size = int(len(shuffled) * factor)
    return data[:train_size], data[train_size:]


def save(data, path):
    import pickle
    pickle.dump(data, open(path, 'wb'))


def load(data, path):
    import pickle
    vocab, max_length = pickle.load(open(path, 'wb'))
    return vocab, max_length


def get_xy(data):
    x = []
    y = []
    for d in data:
        x.append(d.tokens_ids)
        y.append(d.labels_ids)

    return np.array(x), np.array(y)


def get_training_batches(data, batch_size):
    num_batches = int(len(data) / batch_size)
    shuffled = shuffle(data)
    for batch_i in range(num_batches):
        start = batch_i * batch_size
        end = min((batch_i + 1) * batch_size, len(data))

        yield data[start: end]


def train_and_test():
    with tf.Graph().as_default():
        np.random.seed(10)
        tf.set_random_seed(10)

        all_train_data, test_data, vocab, max_length, labels = load_qc_data(FLAGS.data_dir)
        num_classes = len(labels)
        save([vocab, labels, max_length], FLAGS.model_dir + "/params.pkl")
        train_data, valid_data = split(all_train_data, 0.8)

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement, inter_op_parallelism_threads=FLAGS.processors_num,
            intra_op_parallelism_threads=FLAGS.processors_num)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("Initializing Embedding")
            embedding = None
            if FLAGS.embedding_path != "":
                embedding = Word2VecEmbedding(FLAGS.embedding_path, vocab, FLAGS.train_embedding)
            else:
                embedding = RandomEmbedding(len(vocab), FLAGS.embedding_size)

            print("Building nn_model")
            model = RnnTextClassifier(batch_size=FLAGS.batch_size, sentence_length=max_length,
                                      embedding=embedding, cell_layer_size=FLAGS.cell_size,
                                      cell_layer_num=FLAGS.rnn_num,
                                      num_classes=num_classes, lr=FLAGS.learning_rate, lam=FLAGS.lam)
            model.build_network()
            print("Building training operations")
            model.build_train_ops()
            model.summary()

            tf.global_variables_initializer().run()

            valid_x, valid_y = get_xy(valid_data)
            test_x, test_y = get_xy(test_data)
            saver = tf.train.Saver(max_to_keep=1)
            best_vd_accuracy = 0.0
            best_vd_loss = 0.0
            best_tt_accuracy = 0.0
            best_tt_loss = 0.0

            writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/train",
                                           sess.graph)

            print("Start training")
            for epoch in range(FLAGS.num_epochs):
                batches = get_training_batches(train_data, FLAGS.batch_size)
                # Training on batches
                for batch in batches:
                    train_x, train_y = get_xy(batch)

                    step, loss, accuracy, summary = model.train(sess, train_x, train_y, FLAGS.dropout_keep_prob)
                    writer.add_summary(summary, step)
                    print("Training: epoch\t{:g}\tstep\t{:g}\tloss\t{:g}\taccuracy\t{:g}".format(epoch, step, loss,
                                                                                                 accuracy))

                # Evaluate on validation and test set
                vd_step, vd_loss, vd_accuracy, _ = model.step(sess, valid_x, valid_y)
                print("Validation: loss\t{:g}\taccuracy\t{:g}".format(vd_loss, vd_accuracy))
                tt_step, tt_loss, tt_accuracy, _ = model.step(sess, test_x, test_y)
                print("Testing: loss\t{:g}\taccuracy\t{:g}".format(tt_loss, tt_accuracy))

                if vd_accuracy > best_vd_accuracy:
                    best_vd_accuracy = vd_accuracy
                    best_vd_loss = vd_loss
                    best_tt_accuracy = tt_accuracy
                    best_tt_loss = tt_loss
                    print("Saving nn_model")
                    saver.save(sess, FLAGS.model_dir + "/qc_model")

            print("Best Validation: loss\t{:g}\taccuracy\t{:g}".format(best_vd_loss, best_vd_accuracy))
            print("Best Testing: loss\t{:g}\taccuracy\t{:g}".format(best_tt_loss, best_tt_accuracy))


if __name__ == "__main__":
    train_and_test()
