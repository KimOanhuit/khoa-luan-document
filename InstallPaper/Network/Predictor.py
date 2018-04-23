from sklearn.linear_model import LogisticRegression
import random
import networkx as nx
import time

import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

class Predictor(object):
    features = []
    output = []
   
    def readData(self):
        f = open('Dataset/Wiki/FeaturesFull_25%.txt','r')
        lines = f.readlines()
        for line in lines:
            # if line == '\n':
            #     continue
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features.append(splittedLine[12:])
            self.output.append(splittedLine[2])
       
    def trainStep1(self):

        print '*------------------KFold-------------------------------*'
        X = np.array(self.features)
        y = np.array(self.output)

        kf = KFold(n_splits = 10, shuffle = False, random_state = None)
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
            print "Accuracy: ", round(acc,3)
            
            fpr, tpr, _ = roc_curve(y_test, pred)
            roc_auc = auc(fpr, tpr)
            print "AUC/ROC: ", round(roc_auc,3)

            



        

        
        
            


    
    
    


