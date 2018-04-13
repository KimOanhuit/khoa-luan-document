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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
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
            Data = pd.read_csv('Dataset/Wiki/DatasetFullWithText.csv')
            graphx = Creator.Creator()
            # graphx.readWiki()
            # graphx.computeFeatures(Data)
            
            predict = Predictor.Predictor()
            # nodeList = predict.readData()
            # print 'How many node you want to random?'
            # num = int(raw_input())
            # listNodesStart = predict.getRandomNodeList(nodeList,num)
            # print 'How many node you want to create a graph with BFS?'
            # numsNode = int(raw_input())
            # predict.BFS(nodeList,listNodesStart, numsNode)
            predict.train1()
            predict.train2()
            predict.train3()
            predict.train4()
            predict.train5()
            predict.train6()
            predict.train7()
            predict.train8()
            predict.train9()
            predict.train10()
        
            print 'Fold 1:'
            predict.predict1()
            print'------------------------'
            print 'Fold 2:'
            predict.predict2()
            print'------------------------'
            print 'Fold 3:'
            predict.predict3()
            print'------------------------'
            print 'Fold 4:'
            predict.predict4()
            print'------------------------'
            print 'Fold 5:'
            predict.predict5()
            print'------------------------'
            print 'Fold 6:'
            predict.predict6()
            print'------------------------'
            print 'Fold 7:'
            predict.predict7()
            print'------------------------'
            print 'Fold 8:'
            predict.predict8()
            print'------------------------'
            print 'Fold 9:'
            predict.predict9()
            print'------------------------'
            print 'Fold 10:'
            predict.predict10()
        
        elif reply == 2:
            #dataframe = pd.read_csv("Dataset/Wiki/TXT.csv")
            dataframe = pd.read_csv("Dataset/Wiki/DatasetCopy1.csv")
            nlp = Text.Text(dataframe)

            # Lam sach cot "TXT"
            text = nlp.clean_text(dataframe, 'TXT')

            # Add text vao list
            text_list = text.values.tolist()

            # Vector hoa
            vocab = nlp.flatten_words(text_list, get_unique=True)
            feature_extraction = TfidfVectorizer(analyzer='word', min_df=1, ngram_range=(1,2),stop_words='english', vocabulary=vocab)
            X = feature_extraction.fit_transform(dataframe["TXT"].values)
            
            # PCA reduction to 2 dimension
            svd = TruncatedSVD(n_components=2)
            SVD = svd.fit_transform(X)
            # dataframe['firstSVD']
            principalDf = pd.DataFrame(data = SVD, columns = ['firstSVD', 'secondSVD'])
            # finalDf = pd.concat([principalDf, dataframe[['Class']]], axis = 1)
            finalDf = pd.concat([principalDf,dataframe], axis = 1)
            finalDf.to_csv('Dataset/Wiki/test.csv')
        
            # # Split the training data
            # train, test = cross_validation.train_test_split(dataframe, test_size = 0.1, random_state = 150)
            # num_train = len(train) 

            # X_train = SVD[:num_train]
            # X_test = SVD[num_train:]

            # y_train = dataframe["Sign"].values[:num_train]
            # y_test = dataframe["Sign"].values[num_train:]

            # # train classifier
            # svm = SVC(probability=True, kernel='rbf')
            # svm.fit(X_train, y_train)

            # # predict and evaluate predictions
            # predictions = svm.predict(X_test)
            # print accuracy_score(y_test, predictions)
        
        elif reply == 3:
            p = Predict.Predict()
            p.train1()
            p.train2()
            p.train3()
            p.train4()
            p.train5()
            p.train6()
            p.train7()
            p.train8()
            p.train9()
            p.train10()
        
            print 'Fold 1:'
            p.predict1()
            print'------------------------'
            print 'Fold 2:'
            p.predict2()
            print'------------------------'
            print 'Fold 3:'
            p.predict3()
            print'------------------------'
            print 'Fold 4:'
            p.predict4()
            print'------------------------'
            print 'Fold 5:'
            p.predict5()
            print'------------------------'
            print 'Fold 6:'
            p.predict6()
            print'------------------------'
            print 'Fold 7:'
            p.predict7()
            print'------------------------'
            print 'Fold 8:'
            p.predict8()
            print'------------------------'
            print 'Fold 9:'
            p.predict9()
            print'------------------------'
            print 'Fold 10:'
            p.predict10()

        elif reply == 4:
            break

if __name__ == '__main__':     # if the function is the main function ...
    main()