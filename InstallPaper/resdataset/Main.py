from Predictor import Predictor
from Creator import Creator
import networkx as nx
import pandas as pd

def printMenu():
    print "-------------------------------------"
    print "               Menu              "
    print "-------------------------------------"
    print "Choose:"
    print "1. Create Wikipedia Graph full"
    print "2. Predictor"
    print "3. Exit"

    try:
        reply = int(raw_input('Answer:'))
    except ValueError:
        print "Not a number"

    return reply

def main():
    while True:

        reply = printMenu()

        if reply == 1:
            Data = pd.read_csv('../Dataset/WikiDataset.csv')
            graphx = Creator()
            graphx.wikiNetworkx(Data)
            graphx.random_node()
        
        elif reply == 2:
            predict = Predictor()
            predict.readData() 
            predict.deviceTrainTest()
            # predict.train()
            # predict.test()


        elif reply == 3:
            break

if __name__ == '__main__':     # if the function is the main function ...
    main()