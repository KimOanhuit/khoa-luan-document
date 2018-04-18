import re
import networkx as nx
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

class Predict(object):
    information_txt = {}
    feature = {}
    outputs = []
    edges = {}
    
    def Text(self, Data):
        for i in range(0, len(Data)):
            self.information_txt[int(Data.at[i,'SRC']), int(Data.at[i,'TGT'])] = str(Data.at[i,'TXT'])
        
        f = open('Dataset/Wiki/FeaturesFull.txt','r')
        for line in f.readlines():
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(int, splittedLine)
            self.edges[(splittedLine[0], splittedLine[1])] = splittedLine[2]

            self.feature[(splittedLine[0], splittedLine[1])] = splittedLine[12:]
        
        for i in self.edges.keys():
            try:
                SRC = i[0]
                TGT = i[1]
                Sign = self.edges[i]
                TXT = self.information_txt[i]
                
            except KeyError:
                continue

    def combineFeatures(self):
        f = open('Dataset/Wiki/CombineStep1.txt', 'w')
        first = "SRC,TGT,Sign,FFpp,FFpm,FFmp,FFmm,FBpp,FBpm,FBmp,FBmm,BFpp,BFpm,BFmp,BFmm,BBpp,BBpm,BBmp,BBmm,TXT\n"
        f.write(first)
        for key in self.feature.keys():

            print self.feature[key]
            try:
                SRC = float(key[0])
                TGT = float(key[1])
                Sign = float(self.edges[key])
                TXT = self.information_txt[key]
                FFpp = float(self.feature[key][0])
                FFpm = float(self.feature[key][1])
                FFmp = float(self.feature[key][2])
                FFmm = float(self.feature[key][3])
                FBpp = float(self.feature[key][4])
                FBpm = float(self.feature[key][5])
                FBmp = float(self.feature[key][6])
                FBmm = float(self.feature[key][7])
                BFpp = float(self.feature[key][8])
                BFpm = float(self.feature[key][9])
                BFmp = float(self.feature[key][10])
                BFmm = float(self.feature[key][11])
                BBpp = float(self.feature[key][12])
                BBpm = float(self.feature[key][13])
                BBmp = float(self.feature[key][14])
                BBmm = float(self.feature[key][15])
                # print FFpp, FFpm,FFmp,FFmm,FBpp,FBpm,FBmp,FBmm,BFpp,BFpm,BFmp,BFmm,BBpp,BBpm,BBmp,BBmm

                text1 = str(SRC)+ ',' + str(TGT) + ',' + str(Sign) + ','
                text2 = str(FFpp) + ',' + str(FFpm) + ',' + str(FFmp) + ',' + str(FFmm) + ',' + str(FBpp) + ',' + str(FBpm) + ',' + str(FBmp) + ',' + str(FBmm) + ','
                text3 = str(BFpp) + ',' + str(BFpm) + ',' + str(BFmp) + ',' + str(BFmm) + ',' + str(BBpp) + ',' + str(BBpm) + ',' + str(BBmp) + ',' + str(BBmm) + ',' + TXT + '\n'
                txt = text1 + text2 + text3
                f.write(txt)
            except KeyError:
                continue 
        
        f.close()

    def clean_text(self, dataframe, col):
        return dataframe[col].fillna('').apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                    .apply(lambda x: re.sub('\s+', ' ', x).strip())

    def remove_stopwords(self, tokenized_words):
        self.stop_words = stopwords.words('english')
        return [[w.lower() for w in sent
                if (w.lower() not in stop_words)]
                for sent in tokenized_words]
    
    def count_pattern(self, dataframe, col, pattern):
        dataframe = dataframe.copy()
        return dataframe[col].str.count(pattern)

    def split_on_word(self, text):
        if type(text) is list:
            return [regexp_tokenize(sentence, pattern="\w+(?:[-']\w+)*") for sentence in text]
        else:
            return regexp_tokenize(text, pattern="\w+(?:[-']\w+)*")

    def flatten_words(self, list1d, get_unique=False):
        qa = [s.split() for s in list1d]
        if get_unique:
            return sorted(list(set([w for sent in qa for w in sent])))
        else:
            return [w for sent in qa for w in sent]

    def allFeatures(self, dataframe):
        dataframe = dataframe.fillna(' ')
        # Lam sach cot "TXT"
        text = self.clean_text(dataframe, 'TXT')

        # Add text vao list
        text_list = text.values.tolist()
       
        # Vector hoa
        vocab = self.flatten_words(text_list, get_unique=True)
        feature_extraction = TfidfVectorizer(analyzer='word', min_df=1, ngram_range=(1,1),stop_words='english', vocabulary=vocab, max_features=10000)

        mapper = DataFrameMapper([
            ('FFpp', None),
            ('FFpm', None),
            ('FFmp', None),
            ('FFmm', None),
            ('FBpp', None),
            ('FBpm', None),
            ('FBmp', None),
            ('FBmm', None),
            ('BFpp', None),
            ('BFpm', None),
            ('BFmp', None),
            ('BFmm', None),
            ('BBpp', None),
            ('BBpm', None),
            ('BBmp', None),
            ('BBmm', None),
            ('TXT', feature_extraction)
        ])
        X = mapper.fit_transform(dataframe.copy(), 2)
        y = dataframe["Sign"].values

        # kf = KFold(n_splits = 10, shuffle = False, random_state = None)

        # for train, test in kf.split(X):
        #     X_train = X[train]
        #     y_train = y[train]
        #     X_test = X[test]
        #     y_test = y[test]
        
        #     logistic = LogisticRegression()
        #     clf = logistic.fit(X_train, y_train)
        #     pred = logistic.predict(X_test)
                
        #     acc = accuracy_score(pred, y_test)
        #     print "Accuracy: ", round(acc,3)
                
        #     fpr, tpr, _ = roc_curve(y_test, pred)
        #     roc_auc = auc(fpr, tpr)
        #     print 'AUC/ROC: ' + str(roc_auc)
