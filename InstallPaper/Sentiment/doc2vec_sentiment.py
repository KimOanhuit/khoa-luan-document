# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy as np

# shuffle
from random import shuffle

# classifier
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
    
    # Loai bo dau cau ".", "," ...
    def clean_text(self, dataframe, col):
        return dataframe[col].fillna('').apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                    .apply(lambda x: re.sub('\s+', ' ', x).strip())
   
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i, v in enumerate(reviews):
            label = '%s_%s'%(label_type, i)
            labelized.append(LabeledSentence(v, [label]))
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

            #instantiate our DM and DBOW models
            model_dm = Doc2Vec(min_count=3, window=10, size=size, sample=1e-3, negative=5, workers=3)
            model_dbow = Doc2Vec(min_count=3, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

            # build vocab over all reviews
            model_dm.build_vocab(X_train)
            model_dbow.build_vocab(X_train)

            for epoch in range(10):
                perm = np.random.permutation(X_train.shape[0])
                model_dm.train(X_train[perm])
                model_dbow.train(X_train[perm])

            train_vecs_dm = self.getVecs(model_dm, x_train, size)
            train_vecs_dbow = self.getVecs(model_dbow, x_train, size)

            train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
            
    # Get training set vectors from our models
    def getVecs(model, corpus, size):
        vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)




    
            

