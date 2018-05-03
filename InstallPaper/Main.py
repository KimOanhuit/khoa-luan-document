from Network import Predictor
from Network import Creator
from Sentiment import Text
from Sentiment import Doc2VecSentiment
from Combine import Predict

import pandas as pd



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
            Data = pd.read_csv('Dataset/Wiki/DatasetFullTestCopy.csv')
            # graphx = Creator.Creator()
            # graphx.readWiki()
            # graphx.computeFeatures(Data)
            
            predict = Predictor.Predictor()
            predict.readData()
            print '********************************'
            print 'Predicting with Triads Balance Features'
            predict.train()

            print '********************************'
            print 'Predicting with PNR Features'
            predict.train1()
        
        elif reply == 2:
            dataframe = pd.read_csv("Dataset/Wiki/DatasetFullTestCopy.csv")
            # text = Text.Text(dataframe)
            # text.predict(dataframe)
            sentiment = Doc2VecSentiment.Doc2VecSentiment(dataframe)
            sentiment.doc2Vec(dataframe)

        elif reply == 3:
            dataframe = pd.read_csv('Dataset/Wiki/DatasetFullTestCopy.csv')
            new_dataframe = pd.read_csv('Dataset/Wiki/Combine_25%.csv')
            p = Predict.Predict()
            p.Text(dataframe)
            p.combineFeatures()
            p.allFeatures(new_dataframe)
            
        elif reply == 4:
            break

if __name__ == '__main__':     # if the function is the main function ...
    main()