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
from sklearn.metrics import precision_recall_curve_neg
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy import interp
import matplotlib.pyplot as plt

class Predict(object):
    information_txt = {}
    feature = {}
    feature_list = []
    outputs = []
    edges = {}
    X_graph = [[]]
    
    def Text(self, Data):
        for i in range(0, len(Data)):
            self.information_txt[int(Data.at[i,'SRC']), int(Data.at[i,'TGT'])] = str(Data.at[i,'TXT'])
        
        f = open('Dataset/Wiki/FeaturesFull_25%.txt','r')
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
        # f = open('Dataset/Wiki/Combine_25%.txt', 'w')
        # first = "SRC,TGT,Sign,FFpp,FFpm,FFmp,FFmm,FBpp,FBpm,FBmp,FBmm,BFpp,BFpm,BFmp,BFmm,BBpp,BBpm,BBmp,BBmm,TXT\n"
        # f.write(first)
        for key in self.feature.keys():

            # print self.feature[key]
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
                # f.write(txt)
                self.feature_list.append(self.feature[key])
            except KeyError:
                continue 
        self.X_graph = np.asarray(self.feature_list)
        # f.close()

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
        data = dataframe.fillna('')
        # Lam sach cot "TXT"
        text = self.clean_text(dataframe, 'TXT')

        # Add text vao list
        text_list = text.values.tolist()

        # Vector hoa
        vocab = self.flatten_words(text_list, get_unique=True)
        feature_extraction = TfidfVectorizer(analyzer='word', min_df=1, ngram_range=(1,1),stop_words='english', vocabulary=vocab, max_features = 10000)

        X_text = feature_extraction.fit_transform(dataframe["TXT"].values)
        # X_txt = X_text.todense()
        
        # _dataframe = dataframe[['FFpp', 'FFpm', 'FFmp','FFmm','FBpp', 'FBpm', 'FBmp','FBmm','BFpp', 'BFpm', 'BFmp','BFmm','BBpp', 'BBpm', 'BBmp','BBmm']].values.astype(float)
        # print _dataframe
        
        X = hstack([X_text,self.X_graph],format='csr')
        y = dataframe["Sign"].values

        kf = KFold(n_splits = 10, shuffle = False, random_state = None)

        accs = []

        tprs = []
        aucs_roc = []
        mean_fpr = np.linspace(0,1,100)

        aucs_PR = []
        aucs_negPR = []
        
        for train, test in kf.split(X):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
    
            # clf = SVC(kernel='linear', C = 1).fit(X_train, y_train)
            # pred = clf.predict(X_test)
            logistic = LogisticRegression()
            clf = logistic.fit(X_train, y_train)
            pred = logistic.predict(X_test)
            
            acc = accuracy_score(pred, y_test)
            accs.append(acc)
            print "Accuracy: ", round(acc,2)
            
            fpr, tpr, _ = roc_curve(y_test, pred)
            tprs.append(interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs_roc.append(roc_auc)
            print "AUC/ROC: ", round(roc_auc,2)

            precision, recall, thresholds = precision_recall_curve(y_test, pred)
            auc_PR = auc(recall, precision)
            aucs_PR.append(auc_PR)
            print "AUC/PR: ", round(auc_PR,2)

            precision_neg, recall_neg, thresholds_neg = precision_recall_curve_neg(y_test, pred)
            auc_negPR = auc(recall_neg, precision_neg)
            aucs_negPR.append(auc_negPR)
            print "AUC/negPR: ", round(auc_negPR,2)
        
        accs_sum = sum(accs)
        accs_length = len(accs)
        accs_mean = accs_sum / accs_length
        print "Accuracy Average: ", round(accs_mean,2)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        print "AUC/ROC Average: ", round(mean_auc,2)

        aucs_PR_sum = sum(aucs_PR)
        aucs_PR_length = len(aucs_PR)
        aucs_PR_mean = aucs_PR_sum / aucs_PR_length
        print "AUC/PR Average: ", round(aucs_PR_mean,2)

        aucs_negPR_sum = sum(aucs_negPR)
        aucs_negPR_length = len(aucs_negPR)
        aucs_negPR_mean = aucs_negPR_sum / aucs_negPR_length
        print "AUC/negPR Average: ", round(aucs_negPR_mean,2)
        