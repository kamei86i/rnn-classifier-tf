import tensorflow as tf


class RnnTextClassifier:
    def __init__(self, batch_size, sentence_length, embedding, cell_layer_size, cell_layer_num, num_classes, lam=1,
                 lr=0.001):
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.embedding = embedding
        self.cell_layer_size = cell_layer_size
        self.cell_layer_num = cell_layer_num
        self.num_classes = num_classes
        self.dtype = tf.float32
        self.lr = lr
        self.lmd = lam

    def build_network(self):
        with tf.name_scope('input'):
            self.input_x = tf.placeholder(shape=[None, self.sentence_length], dtype=tf.int32, name="input_x")
            self.input_y = tf.placeholder(shape=[None, self.num_classes], dtype=self.dtype, name="input_y")
            self.dropout = tf.placeholder(dtype=self.dtype, name="dropout")

        with tf.name_scope('embedding'):
            # create embedding variable
            emb_w = tf.Variable(initial_value=self.embedding.get_w(), name="w", trainable=self.embedding.is_trainable(),
                                dtype=self.dtype)
            # do embedding lookup
            embedding_input = tf.nn.embedding_lookup(emb_w, self.input_x, name="lookup_op")

        # define the GRU cell
        with tf.name_scope('rnn_cell'):
            cell = tf.nn.rnn_cell.GRUCell(self.cell_layer_size, activation=tf.nn.relu)

            if self.cell_layer_num > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.cell_layer_num)

        # define the RNN operation
        with tf.name_scope('rnn_ops'):
            output, state = tf.nn.dynamic_rnn(cell, embedding_input, time_major=False, dtype=self.dtype)

        to_classify = state
        if self.cell_layer_num > 1:
            to_classify = tf.concat(1, to_classify)

        with tf.name_scope('dropout'):
            to_classify = tf.nn.dropout(to_classify, self.dropout)

        with tf.name_scope('classifier'):
            w = tf.get_variable(name="W", shape=[self.cell_layer_size * self.cell_layer_num, self.num_classes],
                                dtype=self.dtype,
                                initializer=tf.random_uniform_initializer(0, 1, 0))
            b = tf.get_variable(name="b", shape=[self.num_classes], dtype=self.dtype,
                                initializer=tf.constant_initializer(0.1))
            self.l2_loss = tf.nn.l2_loss(w, name="l2_loss")
            scores = tf.nn.xw_plus_b(to_classify, w, b, name="logits")
            self.predictions = tf.argmax(scores, 1, name="predictions")

        with tf.name_scope('loss'):
            losses = self.softmax_cross_entropy(scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + self.lmd * self.l2_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)

    def softmax_cross_entropy(self, scores, gold):
        logsoftmax = tf.log(tf.nn.softmax(scores) + 1e-9)
        return tf.neg(tf.reduce_sum(tf.mul(logsoftmax, gold), 1))

    def summary(self):
        self.merged = tf.summary.merge_all()

    def build_train_ops(self):
        with tf.name_scope('training_operations'):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr, name="Adam")
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step,
                                                           name="train_op")

    def train(self, session, batch_x, batch_y, dropout):
        feed_dict = {
            self.input_x: batch_x,
            self.input_y: batch_y,
            self.dropout: dropout
        }

        _, step, loss, accuracy, summary = session.run(
            [self.train_op, self.global_step, self.loss, self.accuracy, self.merged], feed_dict)
        return step, loss, accuracy, summary

    def step(self, session, x, y):
        feed_dict = {
            self.input_x: x,
            self.input_y: y,
            self.dropout: 1.0
        }
        step, loss, accuracy, predictions = session.run(
            [self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
        return step, loss, accuracy, predictions
