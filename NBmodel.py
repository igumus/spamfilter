'''
@author: Jonathan Zernik

Naive Bayes Classifier for spam filter.
Uses either a boolean value if a word appears
or a NTF (Normalized term frequency) value
for a word in an email.

'''


from bayes_model import Boolean_Model, NTF_Model

import pickle
import os
import re
import math

#########################################


def get_files(path):
    for f in os.listdir(path):
        f = os.path.abspath(os.path.join(path, f))
        if os.path.isfile(f):
            yield f


class NaiveBayesModel:

    def __init__(self, features_file):
        self.features = pickle.load(open(features_file, 'rb'))
        self.model = None

    def pickle(self, model_file):
        output = open(model_file, 'wb')
        pickle.dump(self.model, output)
        print "pickled"

    def test(self, model_file, spam_dir, ham_dir, cost_ratio):
        self.model = pickle.load(open(model_file, 'rb'))
        N = 0
        loss = 0.
        for f in get_files(spam_dir):
            if N < 1000:
                N += 1
                classification = self.classify(self.munge(f), cost_ratio)
                if not (classification == 1):
                    loss += 1

        for f in get_files(ham_dir):
            if N < 2000:
                N += 1
                classification = self.classify(self.munge(f), cost_ratio)
                if not (classification == 0):
                    loss += cost_ratio

        print "Classifier average loss: %f" % (loss / N)


class NB_Boolean(NaiveBayesModel):

    def classify(self, example, cost_ratio):
        log_likelihood1 = math.log(self.model.base_param)
        log_likelihood2 = math.log(1 - self.model.base_param)
        for i, token in enumerate(self.model.attribute_params):
            if example[i] == 1:
                log_likelihood1 += math.log(self.model.attribute_params[i][0])
                log_likelihood2 += math.log(self.model.attribute_params[i][1])
            else:
                log_likelihood1 += math.log(1 -
                                            self.model.attribute_params[i][0])
                log_likelihood2 += math.log(1 -
                                            self.model.attribute_params[i][1])
        return int(log_likelihood1 - math.log(cost_ratio) > log_likelihood2)

    def train(self, spam_dir, ham_dir):
        self.model = Boolean_Model()
        self.model.set_features(self.features)
        N = 0
        loss = 0.
        for f in get_files(spam_dir):
            print N
            N += 1
            self.model.observe_example(self.munge(f), 1)
        for f in get_files(ham_dir):
            print N
            N += 1
            self.model.observe_example(self.munge(f), 0)
        self.model.build_network()
        print "finished training"

    def munge(self, email_file):
        f = open(email_file, 'rb')
        text = f.read()
        word_list = re.split('\W+', text)
        boolean_vector = [int(token in word_list) for token in self.features]
        return boolean_vector


class NB_NTF(NaiveBayesModel):

    def classify(self, example, cost_ratio):
        log_likelihood1 = math.log(self.model.base_param)
        log_likelihood2 = math.log(1 - self.model.base_param)
        for i, token in enumerate(self.model.attribute_params):
            log_likelihood1 += (math.log(self.model.attribute_params[i][0]) -
                                example[i] / self.model.attribute_params[i][0])
            log_likelihood2 += (math.log(self.model.attribute_params[i][1]) -
                                example[i] / self.model.attribute_params[i][1])
        return int(log_likelihood1 - math.log(cost_ratio) > log_likelihood2)

    def train(self, spam_dir, ham_dir):
        self.model = NTF_Model()
        self.model.set_features(self.features)
        N = 0
        loss = 0.
        for f in get_files(spam_dir):
            print N
            N += 1
            self.model.observe_example(self.munge(f), 1)
        for f in get_files(ham_dir):
            print N
            N += 1
            self.model.observe_example(self.munge(f), 0)
        self.model.build_network()
        print "finished training"

    def munge(self, email_file):
        f = open(email_file, 'rb')
        text = f.read()
        word_list = re.split('\W+', text)
        num_words = len(word_list)
        ntf_vector = [float(word_list.count(token)) /
                      num_words for token in self.features]
        return ntf_vector

#########################################


## munge all the words in an email file
def munge_All_Words(email_file):
    f = open(email_file, 'rb')
    text = f.read()
    word_list = re.split('\W+', text)
    new_words_list = list(set(word_list))
    return new_words_list


class Feature_Chooser:
    def __init__(self):
        self.features = []
        self.model = Boolean_Model()
        self.threshold = .03

    def choose(self, spam_dir, ham_dir):
        i = 0
        for f in get_files(spam_dir):
            i += 1
            for word in munge_All_Words(f):
                if word not in self.features:
                    self.features.append(word)
        j = 0
        for f in get_files(ham_dir):
            j += 1
            for word in munge_All_Words(f):
                if word not in self.features:
                    self.features.append(word)
        print len(self.features)
        self.model.set_features(self.features)
        print "finished choosing features"

    def train(self, spam_dir, ham_dir):
        N = 0
        loss = 0.
        for f in get_files(spam_dir):
            N += 1
            if N % 23 == 0:
                print N
                self.model.observe_example(self.munge(f), 1)
        for f in get_files(ham_dir):
            N += 1
            print N
            if N % 23 == 0:
                print N
                self.model.observe_example(self.munge(f), 0)
        self.model.build_network()
        new_features = []
        for i, attribute in enumerate(self.model.attribute_params):
            print attribute, attribute[1] - attribute[0]
            if abs(attribute[1] - attribute[0]) > self.threshold:
                new_features.append(self.features[i])
        print new_features
        self.features = new_features
        print "finished training"

    def munge(self, email_file):
        f = open(email_file, 'rb')
        text = f.read()
        word_list = re.split('\W+', text)
        boolean_vector = [int(token in word_list) for token in self.features]
        return boolean_vector

    def pickle(self, features_file):
        output = open(features_file, 'wb')
        pickle.dump(self.features, output)
        print self.features
        print "pickled"

#########################################
