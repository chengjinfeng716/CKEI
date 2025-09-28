import numpy as np


class VocabDictionary:
    def __init__(self):

        self.UNK_idx = 0
        self.PAD_idx = 1
        self.EOS_idx = 2
        self.SOS_idx = 3
        self.USR_idx = 4
        self.SYS_idx = 5
        self.CLS_idx = 6

        self.unk_token = "UNK"
        self.pad_token = "PAD"
        self.eos_token = "EOS"
        self.sos_token = "SOS"
        self.usr_token = "USR"
        self.sys_token = "SYS"
        self.cls_token = "CLS"

        self.word2index = None
        self.word2count = None
        self.index2word = None
        self.n_words = None

        self.vad = None

    def text2vect(self, text):
        vect = [self.word2index[word] if word in self.word2index else self.UNK_idx for word in text]
        return vect

    def vect2text(self, vect):
        text = [self.index2word[index] if index in self.index2word else self.unk_token for index in vect]
        return " ".join(text)

    def vad_by_word(self, word):
        v, a, d = self.vad[word]
        a = a / 2
        return (np.linalg.norm(np.array([v, a] - np.array([0.5, 0]))) - 0.06467) / 0.607468

    def vad_sentence(self, sentence, pad=-1e18):
        return [self.vad_by_word(word) if word in self.vad else pad for word in sentence]

    def vad_sentence_ids(self, sentence_ids):
        sentence = [self.index2word[index] if index in self.index2word else self.unk_token for index in sentence_ids]
        return self.vad_sentence(sentence, 0)

    def vad_sentence_metric(self, sentence):
        return [self.vad_by_word(word) if word in self.vad else 0 for word in sentence]
