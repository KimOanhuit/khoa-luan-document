from Network import Predictor
from Network import Creator
from Sentiment import Text
from Combine import Predict
import networkx as nx
import pandas as pd

import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def printMenu():
    print "-------------------------------------"
    print "               Menu              "
    print "-------------------------------------"
    print "Choose:"
    print "1. Model Network"
    print "2. Model Sentiment"
    print "3. Model Combine"
    print "4. Exit"

    try:
        reply = int(raw_input('Answer:'))
    except ValueError:
        print "Not a number"

    return reply

def main():
    while True:

        reply = printMenu()

        if reply == 1:
            Data = pd.read_csv('Dataset/Wiki/DatasetFullCopy.csv')
            # graphx = Creator.Creator()
            # graphx.readWiki()
            # graphx.computeFeatures(Data)
            
            predict = Predictor.Predictor()
            predict.readData()
            predict.trainStep1()
        
        elif reply == 2:
            dataframe = pd.read_csv("Dataset/Wiki/DatasetFullCopy.csv")
            text = Text.Text(dataframe)
            text.predict(dataframe)

        elif reply == 3:
            data_original = pd.read_csv('Dataset/Wiki/DatasetFullCopy.csv')
            dataframe = pd.read_csv('Dataset/Wiki/CombineStep2.csv')
            p = Predict.Predict()
            p.Text(data_original)
            # p.combineFeatures()
            p.allFeatures(dataframe)
            # p.readData()
            # p.features(df)

        elif reply == 4:
            break

if __name__ == '__main__':     # if the function is the main function ...
    main()