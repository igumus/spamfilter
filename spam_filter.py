'''
@author: Jonathan Zernik

Command line tool for Naive Bayes
spam filter

'''

import sys
import os
import pickle

from NBmodel import Feature_Chooser, NB_Boolean, NB_NTF

features_file = 'features.pkl'

if __name__ == '__main__':
    def usage():
        sys.stderr.write(
            'Usage: %s -getfeatures\n' %
                         os.path.basename(sys.argv[0]))
        sys.stderr.write(
            'Usage: %s -makemodel [boolean or ntf]\n' %
                         os.path.basename(sys.argv[0]))
        sys.stderr.write(
            'Usage: %s -testmodel [boolean or ntf] [cost_ratio]\n' %
                         os.path.basename(sys.argv[0]))
        sys.stderr.write(
            'Usage: %s -filteremail [boolean or ntf]' \
                ' [filename] [cost_ratio]\n' %
                         os.path.basename(sys.argv[0]))
        sys.exit(1)

    if len(sys.argv) == 2 and sys.argv[1] == '-getfeatures':
        chooser = Feature_Chooser()
        chooser.choose('train/spam', 'train/ham')
        chooser.train('train/spam', 'train/ham')
        chooser.pickle(features_file)
        print "pickled features file"

    elif len(sys.argv) == 3 and sys.argv[1] == '-makemodel':
        if sys.argv[2] == 'boolean':
            model_file = 'boolean_model.pkl'
            mod_gen = NB_Boolean(features_file)
            mod_gen.train('train/spam', 'train/ham')
            mod_gen.pickle(model_file)
            print "pickled boolean model file"
        elif sys.argv[2] == 'ntf':
            model_file = 'ntf_model.pkl'
            mod_gen = NB_NTF(features_file)
            mod_gen.train('train/spam', 'train/ham')
            mod_gen.pickle(model_file)
            print "pickled ntf model file"
        else:
            usage()

    elif len(sys.argv) == 4 and sys.argv[1] == '-testmodel':
        cost_ratio = float(sys.argv[3])
        if sys.argv[2] == 'boolean':
            model_file = 'boolean_model.pkl'
            classifier = NB_Boolean(features_file)
            classifier.test(model_file, 'train/spam', 'train/ham', cost_ratio)
        elif sys.argv[2] == 'ntf':
            model_file = 'ntf_model.pkl'
            classifier = NB_NTF(features_file)
            classifier.test(model_file, 'train/spam', 'train/ham', cost_ratio)
        else:
            usage()

    elif len(sys.argv) == 5 and sys.argv[1] == '-filteremail':
        email_file = str(sys.argv[3])
        cost_ratio = float(sys.argv[4])
        if sys.argv[2] == 'boolean':
            model_file = 'boolean_model.pkl'
            classifier = NB_Boolean(features_file)
            classifier.model = pickle.load(open(model_file, 'rb'))
            print classifier.classify(classifier.munge(email_file), cost_ratio)
        elif sys.argv[2] == 'ntf':
            model_file = 'ntf_model.pkl'
            classifier = NB_NTF(features_file)
            classifier.model = pickle.load(open(model_file, 'rb'))
            print classifier.classify(classifier.munge(email_file), cost_ratio)
        else:
            usage()

    else:
        usage()
