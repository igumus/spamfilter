'''
@author: Jonathan Zernik

Boolean and NTF models for use
in the Naive Bayes Classifier.

'''

import math


class Boolean_Model(object):
    def __init__(self):
        self.base_count = [1, 1]

    def set_features(self, features_list):
        self.attribute_counts = [[[1, 1], [1, 1]] for feature in features_list]

    def observe_example(self, example, is_ham):
        if is_ham == 1:
            self.base_count[0] += 1
            for i, attribute in enumerate(example):
                if attribute == 1:
                    self.attribute_counts[i][0][0] += 1
                else:
                    self.attribute_counts[i][0][1] += 1
        else:
            self.base_count[1] += 1
            for i, attribute in enumerate(example):
                if attribute == 1:
                    self.attribute_counts[i][1][0] += 1
                else:
                    self.attribute_counts[i][1][1] += 1

    def build_network(self):
        self.base_param = float(self.base_count[0]) / sum(self.base_count)
        self.attribute_params = [[float(count[0][0]) / sum(count[0]),
                                  float(count[1][0]) / sum(count[1])]
                                 for count in self.attribute_counts]
        self.base_count = None
        self.attribute_counts = None


class NTF_Model(object):
    def __init__(self):
        self.base_count = [1, 1]

    def set_features(self, features_list):
        self.attribute_counts = [[[1], [1]] for feature in features_list]

    def observe_example(self, example, is_ham):
        if is_ham == 1:
            self.base_count[0] += 1
            for i, attribute in enumerate(example):
                self.attribute_counts[i][0].append(example[i])
        else:
            self.base_count[1] += 1
            for i, attribute in enumerate(example):
                self.attribute_counts[i][1].append(example[i])

    def build_network(self):
        self.base_param = float(self.base_count[0]) / sum(self.base_count)
        self.attribute_params = [[sum(count[0], 0.0) / len(count[0]),
                                  sum(count[1], 0.0) / len(count[1])]
                                 for count in self.attribute_counts]
        self.base_count = None
        self.attribute_counts = None
