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
from sklearn.metrics import precision_recall_curve_neg
from scipy import interp
import matplotlib.pyplot as plt

class Predictor(object):
    features = []
    output = []
    pnr_features = []
   
    def readData(self):
        f = open('Dataset/Wiki/FeaturesFull_12.5%.txt','r')
        lines = f.readlines()
        for line in lines:
            # if line == '\n':
            #     continue
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features.append(splittedLine[5:])
            self.pnr_features.append(splittedLine[3:])
            self.output.append(splittedLine[2])
       
    def train(self):

        print '*------------------KFold-------------------------------*'
        X = np.array(self.features)
        y = np.array(self.output)

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

    def train1(self):

        print '*------------------KFold-------------------------------*'
        X = np.array(self.pnr_features)
        y = np.array(self.output)

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
    
            clf = SVC(kernel='linear', C = 1).fit(X_train, y_train)
            pred = clf.predict(X_test)
            # logistic = LogisticRegression()
            # clf = logistic.fit(X_train, y_train)
            # pred = logistic.predict(X_test)
            
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



        

        
        
            


    
    
    


