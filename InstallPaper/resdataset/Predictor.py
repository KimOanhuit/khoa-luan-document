from sklearn.linear_model import LogisticRegression
import random
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from Creator import Creator

class Predictor(object):
    features = []
    featuresOther = []
    output = []
    outputOther = []
    logistic = []
    trainThreshold = 0
    graph = nx.DiGraph()
    edgeSignDict = {} 

    # def readData(self):
    #     f = open('../Dataset/Features_copy.txt','r')  
    #     for line in f.readlines():
    #         line.replace('\n',",")
    #         splittedLine = line.split(",")
    #         # splittedLine[-1] = "".join(splittedLine[-1].split("\n"))
    #         splittedLine = map(float, splittedLine)
    #         selectedList = splittedLine[3:]
    #         self.features.append(selectedList)
    #         pair1 = str(splittedLine[0]) + "," + str(splittedLine[1])
    #         pair2 = str(splittedLine[1]) + "," + str(splittedLine[0])
    #         if pair1 not in self.edgeSignDict:
    #             self.edgeSignDict[pair1] = splittedLine[2]
    #         else:
    #             if pair2 not in self.edgeSignDict:
    #                 self.edgeSignDict[pair1] += splittedLine[2]
            

        # self.createGraph()
        # f.close()
        # self.readData1(True,"/home/thanos/graphs/all-new")

    
    def readData(self):
        f = open('../Dataset/Features_copy.txt','r')  
        for line in f.readlines():
            line.replace('\n',",")
            splittedLine = line.split(",")
            # splittedLine = map(float, splittedLine)
            selectedList = splittedLine[3:]
            print splittedLine[0]
            # self.features.append(selectedList)

            pair1 = str(splittedLine[0]) + "," + str(splittedLine[1])
            pair2 = str(splittedLine[1]) + "," + str(splittedLine[0])
            if pair1 not in self.edgeSignDict:
                self.edgeSignDict[pair1] = splittedLine[2]
            else:
                if pair2 not in self.edgeSignDict:
                    self.edgeSignDict[pair1] += splittedLine[2]

    def train(self):
        self.trainThreshold = int(round(len(self.features)*0.8))
        self.logistic = LogisticRegression()
        self.logistic.fit(self.features[:self.trainThreshold],self.output[:self.trainThreshold])

    def predict(self):
        print self.features[self.trainThreshold:]
        predictedSigns = self.logistic.predict(self.features[self.trainThreshold:])
        list1 = self.logistic.predict_proba(self.features[self.trainThreshold:])
        self.computeAccuracy(list1,predictedSigns)

    def computeAccuracy(self,predictions,predictedSigns):
        out = self.output[self.trainThreshold:]
        correct = 0
        for i in range(0,len(out)):
            if out[i] == predictedSigns[i]:
                correct+=1

        accuracy = correct / (1.0*len(out))
        print "Accuracy: " + str(accuracy)

    # def createGraph(self):
    #     one = 0
    #     two = 0
    #     three = 0
    #     for edge, sign in self.edgeSignDict.iteritems():

    #         node1 = int(float(edge.split(",")[0]))
    #         node2 = int(float(edge.split(",")[1]))
    #         if int(sign) < 0:
    #             one +=1
    #             self.graph.add_edge(node1,node2,weight=-1)
    #         elif int(sign) > 0:
    #             two +=1
    #             self.graph.add_edge(node1,node2,weight=1)
    #         else:
    #             three +=1
    #             self.graph.add_edge(node1,node2,weight=0)


    # def calculateStability(self):
    #     balanceTriangle = 0
    #     totalTriangles = 0
    #     for edge,sign in self.edgeSignDict.iteritems():
    #         node1 = int(float(edge.split(",")[0]))
    #         node2 = int(float(edge.split(",")[1]))
    #         commonNeigh = sorted(nx.common_neighbors(self.graph,node1,node2))

    #         for inode in commonNeigh:
    #             sign1n = self.graph.get_edge_data(node1,inode,default={'weight':10})['weight']
    #             sign2n = self.graph.get_edge_data(node2,inode,default={'weight':10})['weight']
    #             sign12 = self.graph.get_edge_data(node1,node2,default={'weight':10})['weight']
    #             mul = sign1n*sign2n*sign12

    #             if mul > 0 and mul <10 :
    #                 balanceTriangle +=1
    #             #if (sign1n*sign2n*sign12) != 0:
    #             totalTriangles += 1

    #     print "Balance percentage: " + str((1.0*balanceTriangle)/totalTriangles)




# if __name__ == '__main__':

#     p = Predictor("/home/thanos/graphs/epinions")
#     p.readData(True)
#     p.train()
#     p.predict()
#     p.calculateStability()
#     print "-----"
