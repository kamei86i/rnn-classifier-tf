from abc import ABCMeta, abstractmethod
import numpy as np
from gensim.models import Word2Vec


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


class Word2VecEmbedding(Embedding):
    def __init__(self, embedding_path, vocab, is_trainable):
        self.embedding_path = embedding_path
        self.vocab = vocab
        self.train_embeddings = is_trainable
        self.rows = len(vocab)

        w2v = Word2Vec.load_word2vec_format(fname=embedding_path, binary=True)
        self.columns = w2v.vector_size
        self.w = np.random.uniform(-1.0, 1.0, size=((self.rows, self.columns))).astype("float32")
        for word in vocab:
            index = vocab[word]
            if word in w2v:
                self.w[index] = w2v[word]




    def get_w(self):
        return self.w

    def is_trainable(self):
        return self.train_embeddings