from abc import ABCMeta, abstractmethod
import numpy as np


class Embedding(metaclass=ABCMeta):
    @abstractmethod
    def get_w(self):
        pass

    @abstractmethod
    def is_trainable(self):
        pass


class RandomEmbedding(Embedding):
    def __init__(self, vocab_size, embedding_size):
        self.rows = vocab_size
        self.columns = embedding_size
        self.train_embeddings = True
        self.w = np.random.uniform(-1.0, 1.0, size=((self.rows, self.columns))).astype("float32")

    def get_w(self):
        return self.w

    def is_trainable(self):
        return self.train_embeddings
