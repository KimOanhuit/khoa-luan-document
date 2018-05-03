# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy as np
import re

# shuffle
from random import shuffle

# classifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# random, itertools, matplotlib
import random
import itertools
import matplotlib.pyplot as plt

class Doc2VecSentiment(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def __iter__(self, reviews, label_type):
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            yield LabeledSentence(v.split(), [label])

    # Loai bo dau cau ".", "," ...
    def clean_text(self, dataframe, col):
        return dataframe[col].fillna('').apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                    .apply(lambda x: re.sub('\s+', ' ', x).strip())
    
    def labelizeReviews(self, reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v.split(), [label]))
        return labelized
    
    def doc2Vec(self, dataframe):
        # Lam sach cot "TXT"
        text = self.clean_text(dataframe, 'TXT')

        # Add text vao list
        text_list = text.values.tolist()
 
        X = dataframe['TXT'].values
        y = dataframe['Sign'].values

        size = 400
        kf = KFold(n_splits = 10, shuffle = False, random_state = None)

        for train, test in kf.split(X):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            X_train = self.labelizeReviews(X_train, 'train')
            X_test = self.labelizeReviews(X_test, 'test')
           
            # instantiate models
            model = Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)

            # build vocab
            model.build_vocab(X_train)
            
            model.train(X_train, total_examples=model.corpus_count, epochs=1)
            print model.most_similar('good')



    
            

