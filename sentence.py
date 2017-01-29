class Sentence:
    def __init__(self, sentence, label, tokens):
        self.sentence = sentence
        self.label = label
        self.tokens = tokens
        self.tokens_ids = []
        self.labels_ids = []

    def pad_to(self, length, pad_token):
        diff = length - len(self.tokens)
        new_tokens = [pad_token] * diff
        self.tokens = new_tokens + self.tokens

        #print(self.tokens)

    def apply_vocabs(self, vocab, unk, labels):
        label_size = len(labels)
        for token in self.tokens:
            if token in vocab:
                self.tokens_ids.append(vocab[token])
            else:
                self.tokens_ids.append(vocab[unk])
            self.labels_ids = [0]*label_size
            self.labels_ids[labels[self.label]] = 1

        #print(len(self.tokens_ids))
        #print(self.tokens_ids)
