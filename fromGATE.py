import numpy as np
import math
from twokenize import tokenize  # for temp use


class GateExporter:
    def __init__(self, export_file, p):
        self. gate_pos = {}
        self.gate_token = {}
        for t, i in zip(export_file, p):
            self.gate_token[i] = tokenize(t)

    def tokenize(self, id):
        return self.gate_token[id]

    def pos(self, id):
        return self.gate_pos[id]

    def get_length(self, id):
        return len(self.gate_token[id])

    def get_avg_word_length(self, id):
        return np.mean([len(word) for word in self.gate_token[id]])

    def get_length_feature(self, ids):
        """
        length feature can make 3 more samples correct
        :param ids:
        :return: numpy appendable feature list
        """
        encoded_list = []
        for id in ids:
            l = self.get_length(id)
            wl = self.get_avg_word_length(id)
            encoded_list.append([math.log2(l), wl])
        return encoded_list
