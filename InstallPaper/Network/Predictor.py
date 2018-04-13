from sklearn.linear_model import LogisticRegression
from random import randrange
import random
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from Creator import Creator

class Predictor(object):
    features1 = []
    output1 = []
    features2 = []
    output2 = []
    features3 = []
    output3 = []
    features4 = []
    output4 = []
    features5 = []
    output5 = []
    features6 = []
    output6 = []
    features7 = []
    output7 = []
    features8 = []
    output8 = []
    features9 = []
    output9 = []
    features10 = []
    output10 = []
    logistic1 = []
    logistic2 = []
    logistic3 = []
    logistic4 = []
    logistic5 = []
    logistic6 = []
    logistic7 = []
    logistic8 = []
    logistic9 = []
    logistic10 = []

    def readData(self):
        f = open('Dataset/Wiki/FeaturesFull_copy.txt','r')
        nodeList = []
        for line in f.readlines():
            line.replace('\n',",")            
            splittedLine = line.split(",")
            nodeList.append(splittedLine)   
        return nodeList
    
    def getRandomNodeList(self,dataList,num):
        randomList = []
        numsOfEdge = len(dataList)
        while len(randomList) < num:
            random_index = randrange(0,numsOfEdge)
            if dataList[random_index][0] not in randomList:
                randomList.append(dataList[random_index][0])
        return randomList

    def BFS(self,dataList,listNodeStart,numsNodeLimit):
        # i = 0
        # for nodeStart in listNodeStart:
        #     i += 1
        #     listNodeSecond = []
        #     checkNode = []
        #     numsNode = 0
            
        #     for node in dataList:
        #         if node in checkNode:
        #             continue
        #         if node[0] == nodeStart or node[1] == nodeStart :
        #             numsNode += 1
        #             if numsNode > numsNodeLimit:
        #                 break
        #             if node[0] not in listNodeSecond:
        #                 listNodeSecond.append(node[0])
        #             if node[1] not in listNodeSecond:
        #                 listNodeSecond.append(node[1])
        #             f = open("Dataset/Wiki/SetBFS/set" + str(i) + ".txt",'a+')
        #             f.write(node[0]+","+node[1]+","+node[2]+","+node[3]+","+node[4]+","+node[5]+","+node[6]+","+node[7]+","+node[8]+","+node[9]+","+node[10]+","+node[11]+","+node[12]+","+node[13]+","+node[14]+","+node[15]+","+node[16]+","+node[17]+","+node[18])            
        #             f.close()
            
        #     for node in dataList:
        #         if node in checkNode:
        #             continue
        #         if node[0] in listNodeSecond or node[1] in listNodeSecond:
        #             numsNode += 1
        #             if numsNode > numsNodeLimit:
        #                 break
        #             if node[0] not in listNodeSecond:
        #                 listNodeSecond.append(node[0])
        #             if node[1] not in listNodeSecond:
        #                 listNodeSecond.append(node[1])
        #             f = open("Dataset/Wiki/SetBFS/set" + str(i) + ".txt",'a+')
        #             f.write(node[0]+","+node[1]+","+node[2]+","+node[3]+","+node[4]+","+node[5]+","+node[6]+","+node[7]+","+node[8]+","+node[9]+","+node[10]+","+node[11]+","+node[12]+","+node[13]+","+node[14]+","+node[15]+","+node[16]+","+node[17]+","+node[18])            
        #             f.close()


        #----------------------------------------------------------------------------------#

        i = 0
        for nodeStart in listNodeStart:
            i += 1
            listNodeSecond = []
            checkNode = []
            numsNode = 0
            
            for node in dataList:
                if node in checkNode:
                    continue
                if node[0] == nodeStart or node[1] == nodeStart :
                    numsNode += 1
                    if numsNode > numsNodeLimit:
                        break
                    if node[0] not in listNodeSecond:
                        listNodeSecond.append(node[0])
                    if node[1] not in listNodeSecond:
                        listNodeSecond.append(node[1])
                    f = open("Dataset/Wiki/SetBFS/set" + str(i) + ".txt",'a+')
                    f.write(node[0]+","+node[1]+","+node[2]+","+node[3]+","+node[4]+","+node[14]+","+node[15]+","+node[16]+","+node[17]+","+node[18]+","+node[19]+","+node[20]+","+node[21]+","+node[22]+","+node[23]+","+node[24]+","+node[25]+","+node[26]+","+node[27]+","+node[28]+","+node[29])            
                    f.close()
            
            for node in dataList:
                if node in checkNode:
                    continue
                if node[0] in listNodeSecond or node[1] in listNodeSecond:
                    numsNode += 1
                    if numsNode > numsNodeLimit:
                        break
                    if node[0] not in listNodeSecond:
                        listNodeSecond.append(node[0])
                    if node[1] not in listNodeSecond:
                        listNodeSecond.append(node[1])
                    f = open("Dataset/Wiki/SetBFS/set" + str(i) + ".txt",'a+')
                    f.write(node[0]+","+node[1]+","+node[2]+","+node[3]+","+node[4]+","+node[14]+","+node[15]+","+node[16]+","+node[17]+","+node[18]+","+node[19]+","+node[20]+","+node[21]+","+node[22]+","+node[23]+","+node[24]+","+node[25]+","+node[26]+","+node[27]+","+node[28]+","+node[29])            
                    f.close()
        
    def train1(self):
        f = open('Dataset/Wiki/SetBFS/set1.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features1.append(splittedLine[5:])
            self.output1.append(splittedLine[2])

        self.logistic1 = LogisticRegression()
        self.logistic1.fit(self.features1,self.output1)

    def train2(self):
        f = open('Dataset/Wiki/SetBFS/set2.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features2.append(splittedLine[5:])
            self.output2.append(splittedLine[2])
        
        self.logistic2 = LogisticRegression()
        self.logistic2.fit(self.features2,self.output2)
    
    def train3(self):
        f = open('Dataset/Wiki/SetBFS/set3.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features3.append(splittedLine[5:])
            self.output3.append(splittedLine[2])
        
        self.logistic3 = LogisticRegression()
        self.logistic3.fit(self.features3,self.output3)
    
    def train4(self):
        f = open('Dataset/Wiki/SetBFS/set4.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features4.append(splittedLine[5:])
            self.output4.append(splittedLine[2])
        
        self.logistic4 = LogisticRegression()
        self.logistic4.fit(self.features4,self.output4)
    
    def train5(self):
        f = open('Dataset/Wiki/SetBFS/set5.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features5.append(splittedLine[5:])
            self.output5.append(splittedLine[2])
        
        self.logistic5 = LogisticRegression()
        self.logistic5.fit(self.features5,self.output5)

    def train6(self):
        f = open('Dataset/Wiki/SetBFS/set6.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features6.append(splittedLine[5:])
            self.output6.append(splittedLine[2])
        
        self.logistic6 = LogisticRegression()
        self.logistic6.fit(self.features6,self.output6)
    
    def train7(self):
        f = open('Dataset/Wiki/SetBFS/set7.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features7.append(splittedLine[5:])
            self.output7.append(splittedLine[2])
        
        self.logistic7 = LogisticRegression()
        self.logistic7.fit(self.features7,self.output7)
    
    def train8(self):
        f = open('Dataset/Wiki/SetBFS/set8.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features8.append(splittedLine[5:])
            self.output8.append(splittedLine[2])
        
        self.logistic8 = LogisticRegression()
        self.logistic8.fit(self.features8,self.output8)

    def train9(self):
        f = open('Dataset/Wiki/SetBFS/set9.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features9.append(splittedLine[5:])
            self.output9.append(splittedLine[2])
        
        self.logistic9 = LogisticRegression()
        self.logistic9.fit(self.features9,self.output9)
    
    def train10(self):
        f = open('Dataset/Wiki/SetBFS/set10.txt','r')
        lines = f.readlines()
        for line in lines:
            line.replace("\n", ",")
            splittedLine = line.split(",")
            splittedLine = map(float, splittedLine)
            self.features10.append(splittedLine[5:])
            self.output10.append(splittedLine[2])
        
        self.logistic10 = LogisticRegression()
        self.logistic10.fit(self.features10,self.output10)
    
    def computeAccuracy(self,out,predictions,predictedSigns):
        correct = 0
        for i in range(0,len(out)):
            if out[i] == predictedSigns[i]:
                correct += 1

        accuracy = correct / (1.0*len(out))
        print "Accuracy: " + str(accuracy)

    def predict1(self):
        predictedSigns = self.logistic1.predict(self.features2)
        list1 = self.logistic1.predict_proba(self.features2)
        self.computeAccuracy(self.output1,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output2,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))   
    
    def predict2(self):
        predictedSigns = self.logistic2.predict(self.features3)
        list1 = self.logistic2.predict_proba(self.features3)
        self.computeAccuracy(self.output2,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output3,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))

    def predict3(self):
        predictedSigns = self.logistic3.predict(self.features4)
        list1 = self.logistic3.predict_proba(self.features4)
        self.computeAccuracy(self.output3,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output4,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))

    def predict4(self):
        predictedSigns = self.logistic4.predict(self.features5)
        list1 = self.logistic4.predict_proba(self.features5)
        self.computeAccuracy(self.output4,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output5,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))

    def predict5(self):
        predictedSigns = self.logistic5.predict(self.features6)
        list1 = self.logistic5.predict_proba(self.features6)
        self.computeAccuracy(self.output5,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output6,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))

    def predict6(self):
        predictedSigns = self.logistic6.predict(self.features7)
        list1 = self.logistic6.predict_proba(self.features7)
        self.computeAccuracy(self.output6,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output7,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))

    def predict7(self):
        predictedSigns = self.logistic7.predict(self.features8)
        list1 = self.logistic7.predict_proba(self.features8)
        self.computeAccuracy(self.output7,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output8,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))

    def predict8(self):
        predictedSigns = self.logistic8.predict(self.features9)
        list1 = self.logistic8.predict_proba(self.features9)
        self.computeAccuracy(self.output8,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output9,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))
    
    def predict9(self):
        predictedSigns = self.logistic9.predict(self.features10)
        list1 = self.logistic9.predict_proba(self.features10)
        self.computeAccuracy(self.output9,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output10,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))
    
    def predict10(self):
        predictedSigns = self.logistic10.predict(self.features1)
        list1 = self.logistic10.predict_proba(self.features1)
        self.computeAccuracy(self.output10,list1,predictedSigns)
        fpr, tpr, _ = roc_curve(self.output1,predictedSigns)
        print 'AUC: ' + str(auc(fpr, tpr))


    
    
    


