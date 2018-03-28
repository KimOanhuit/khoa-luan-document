# from GraphCreator import GraphCreator
from Networkx import Wiki
import networkx as nx
import pandas as pd

def printMenu():
    print "-------------------------------------"
    print "               Menu              "
    print "-------------------------------------"
    print "Choose:"
    print "1. Create Wikipedia Graph full"
    print "2. Create Wikipedia Subgraph with BFS"
    print "3. Exact Triads Balance Featers"
    print "4. Training"
    print "5. Exit"

    try:
        reply = int(raw_input('Answer:'))
    except ValueError:
        print "Not a number"

    return reply

def main():
    while True:
        reply = printMenu()

        if reply == 1:
            # gCreator = GraphCreator()
            # gCreator.readWiki()
            # gCreator.randomNodes()
            # gCreator.computeBFS()
            Data = pd.read_csv('../Dataset/WikiDataset.csv')
            graphx = Wiki()
            graphx.wikiNetworkx(Data)
            graphx.computeFeatures(Data)

        elif reply == 5:
            break

if __name__ == '__main__':     # if the function is the main function ...
    main()

