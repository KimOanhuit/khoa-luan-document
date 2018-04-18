import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import random
from collections import deque

class Creator(object):

    # Data = pd.read_csv('Dataset/WikiDataset.csv')
    wikiGraph = None

    def __init__(self):
        self.wikiGraph = nx.DiGraph()

    def readWiki(self):
        f = open('Dataset/Wiki/WikiDataset.txt','r')
        i = 0
        authorToInt = {}
        num = 0
        source = -1
        target = -1
        information = {}
        txt = []

        for line in f:
            if len(line) < 2:
                source = -1
                target = -1
                i = 0
                continue
            line = line.replace('\n','')
            value = line.split(":")[1]
            i += 1
        
            if i == 1:
                if value not in authorToInt:
                    authorToInt[value] = num
                    num += 1
                self.wikiGraph.add_node(authorToInt[value])
                source = authorToInt[value]

            elif i == 2:
                if value not in authorToInt:
                    authorToInt[value] = num
                    num +=1
                self.wikiGraph.add_node(authorToInt[value])
                target = authorToInt[value]
        
            elif i == 3:
                self.wikiGraph.add_edge(source,target,weight = int(value))
        
            elif i == 7:
                information[(source,target)] = value
                txt.append(value)

            elif i == 8:
                source = -1
                target = -1
                i = 0

        # nx.write_graphml(self.wikiGraph, "Dataset/Wiki/WikiGraph.graphml")

        # edges = self.wikiGraph.edges()
        # f = open('Dataset/Wiki/DatasetFull.txt', 'w')
        # f.write('SRC,TGT,Sign,TXT\n')
        # for edge in edges:
        #     node1 = edge[0]
        #     node2 = edge[1]
        #     sign = self.wikiGraph.get_edge_data(node1, node2)
        #     txt = information[edge].replace(",", "").replace(".", "").replace(":", "").replace(";", "").replace(":", "").replace("--", "").replace("'''", "").replace("''", "")
        #     data = str(node1) + ',' + str(node2) + ',' + str(sign['weight']) + ',' + txt + '\n'
        #     f.write(data)
        
        # f.close()

    def getGraph(self):
        return self.wikiGraph()

    def computeTriads(self, commonNeighbor, node1, node2, triadList, graph):
        tmSign = graph.get_edge_data(node1,commonNeighbor,default={'weight':0})
        tmpSign = tmSign['weight']
        if tmpSign < 0:
            #Fm
            tmSign = graph.get_edge_data(node2,commonNeighbor,default={'weight':0})
            tmpSign = tmSign['weight']
            if tmpSign < 0:
                #FBmm
                triadList[7] += 1
            elif tmpSign > 0:
                #FBmp
                triadList[6] += 1
            else:
                tmSign = graph.get_edge_data(commonNeighbor,node2,default={'weight':0})
                tmpSign = tmSign['weight']
                if tmpSign < 0:
                    #FFmm
                    triadList[3] += 1
                elif tmpSign > 0:
                    #FFmp
                    triadList[2] += 1
        elif tmpSign > 0:
            #Fp
            tmSign = graph.get_edge_data(node2,commonNeighbor,default={'weight':0})
            tmpSign = tmSign['weight']
            #print "Sign:" + str(tmpSign)
            if tmpSign < 0:
                #FBpm
                triadList[5] += 1
            elif tmpSign > 0:
                #FBpp
                triadList[4] += 1
            else:
                tmSign = graph.get_edge_data(commonNeighbor,node2,default={'weight':0})
                tmpSign = tmSign['weight']
                if tmpSign < 0:
                    #FFpm
                    triadList[1] += 1
                elif tmpSign > 0:
                    #FFpp
                    triadList[0] += 1
        else:
            tmSign = graph.get_edge_data(commonNeighbor,node1,default={'weight':0})
            tmpSign = tmSign['weight']
            if tmpSign < 0:
                #Bm
                tmSign = graph.get_edge_data(node2,commonNeighbor,default={'weight':0})
                tmpSign = tmSign['weight']
                if tmpSign < 0:
                    #BBmm
                    triadList[15] += 1
                elif tmpSign > 0:
                    #BBmp
                    triadList[14] += 1
                else:
                    tmSign = graph.get_edge_data(commonNeighbor,node2,default={'weight':0})
                    tmpSign = tmSign['weight']
                    if tmpSign < 0:
                        #BFmm
                        triadList[11] += 1
                    elif tmpSign > 0:
                        #BFmp
                        triadList[10] += 1
            elif tmpSign > 0:
                #Bp
                tmSign = graph.get_edge_data(node2,commonNeighbor,default={'weight':0})
                tmpSign = tmSign['weight']
                if tmpSign < 0:
                    #BBpm
                    triadList[13] += 1
                elif tmpSign > 0:
                    #BBpp
                    triadList[12] += 1
                else:
                    tmSign = graph.get_edge_data(commonNeighbor,node2,default={'weight':0})
                    tmpSign = tmSign['weight']
                    if tmpSign < 0:
                        #BFpm
                        triadList[9] += 1
                    elif tmpSign > 0:
                        #BFpp
                        triadList[8] += 1
        # print "IN: " + str(triadList)
    
    def BFSGraph(self, graph, inode, num, information_txt):
        BFS_listEdges = list(nx.bfs_edges(self.wikiGraph, inode))
        BFS_listNode = []
        for edge in BFS_listEdges:
            if len(BFS_listNode) < num:
                if edge[0] not in BFS_listNode:
                    BFS_listNode.append(edge[0])
                    graph.add_node(edge[0])
                if edge[1] not in BFS_listNode:
                    BFS_listNode.append(edge[1])
                    graph.add_node(edge[1])
            else:
                continue
        
        for node in BFS_listNode:
            in_edge = self.wikiGraph.in_edges(node)
            for edge in in_edge:
                if edge[0] in BFS_listNode and edge[1] in BFS_listNode and edge not in graph.edges:
                    sign = self.wikiGraph.get_edge_data(edge[0], edge[1])
                    graph.add_edge(edge[0], edge[1], weight = int(sign['weight']))
            
            out_edge = self.wikiGraph.out_edges(node)
            for edge in in_edge:
                if edge[0] in BFS_listNode and edge[1] in BFS_listNode and edge not in graph.edges:
                    sign = self.wikiGraph.get_edge_data(edge[0], edge[1])
                    graph.add_edge(edge[0], edge[1], weight = int(sign['weight']))

        f = open('Dataset/Wiki/FeaturesFullStep2.txt','a+')
    
        # first = "SRC,TGT,Sign,firstSVD,secondSVD,In+1,In-1,Out+1,Out-1,In+2,In-2,Out+2,Out-2,CommonNeighbors,"
        # second = "FFpp,FFpm,FFmp,FFmm,FBpp,FBpm,FBmp,FBmm,BFpp,BFpm,BFmp,BFmm,BBpp,BBpm,BBmp,BBmm\n"
        # firstLine = first + second
        # f.write(firstLine)

        edges = graph.edges
        unG = graph.to_undirected(reciprocal=False)

        for edge in edges:
            try:
                test1 = edge[0]
                test2 = edge[1]
                if test1 == test2:
                    continue
                sign = graph.get_edge_data(test1,test2)
                f.write(str(test1) + "," + str(test2) + "," + str(sign['weight']) + ",")
                
                neighbors1 = nx.all_neighbors(graph,test1)
                neighbors2 = nx.all_neighbors(graph,test2)

                commonNeighbors = 0
                nodesSeen = {}
                plusCountIn = 0
                minusCountIn = 0
                plusCountOut = 0
                minusCountOut = 0

                rplusCountIn = 0
                rminusCountIn = 0
                rplusCountOut = 0
                rminusCountOut = 0

                triadList = [0]*16

                for neighbor1 in neighbors1:
                    for neighbor2 in neighbors2:
                        pn = str(test2) + "," + str(neighbor2)
                        if pn not in nodesSeen:
                            nodesSeen[pn] = 1
                            tmSign = self.wikiGraph.get_edge_data(test2,neighbor2,default={'weight':0})
                            tmpSign = tmSign['weight']
                            if tmpSign < 0:
                                rminusCountOut += 1
                            elif tmpSign > 0:
                                rplusCountOut += 1
                            else:
                                tmSign = self.wikiGraph.get_edge_data(neighbor2,test2,default={'weight':0})
                                tmpSign = tmSign['weight']
                                if tmpSign < 0:
                                    rminusCountIn += 1
                                elif tmpSign > 0:
                                    rplusCountIn += 1

                    pn1 = str(test1) + "," + str(neighbor1)
                    if pn1 not in nodesSeen:
                        nodesSeen[pn1] = 1
                        tmSign = graph.get_edge_data(test1,neighbor1,default={'weight':0})
                        tmpSign = tmSign['weight']
                        if tmpSign < 0:
                            minusCountOut += 1
                        elif tmpSign > 0:
                            plusCountOut += 1
                        else:
                            tmSign = graph.get_edge_data(neighbor1,test1,default={'weight':0})
                            tmpSign = tmSign['weight']
                            if tmpSign < 0:
                                minusCountIn += 1
                            elif tmpSign > 0:
                                plusCountIn += 1
                commonNeigh = sorted(nx.common_neighbors(unG,test1,test2))
                for inode in commonNeigh:
                    self.computeTriads(inode,test1,test2,triadList,graph)

                commonNeighbors = len(commonNeigh)

                text1 = str(plusCountIn) + "," + str(minusCountIn) + "," + str(plusCountOut) + "," + str(minusCountOut) + ","
                text2 = str(rplusCountIn) + "," + str(rminusCountIn) + "," + str(rplusCountOut) + "," + str(rminusCountOut) + ","
                final = text1 + text2 + str(commonNeighbors)
                for item in triadList:
                    final += ',' + str(item)

                f.write(final+"\n")
            except KeyError:
                continue
        
        # f.write('\n')
        f.close()
        
        # nx.write_graphml(graph, "Dataset/Wiki/GraphBFS.graphml")
        
    def computeFeatures(self,Data):
        information_txt = {}
        for i in range(0, len(Data)):
            # information_txt[int(Data.at[i,'SRC']), int(Data.at[i,'TGT'])] = (str(Data.at[i,'firstSVD']), str(Data.at[i,'secondSVD']))
            information_txt[int(Data.at[i,'SRC']), int(Data.at[i,'TGT'])] = str(Data.at[i,'TXT'])

        nodes = self.wikiGraph.nodes()
        list_random = [0]*10
        for i in range (0, 10):
            list_random[i] = random.choice(list(nodes))
        
        print 'How many node do you add in BFSGraph?'
        num = int(raw_input())

        for i in list_random:
            self.BFSGraph(nx.DiGraph(), i, num, information_txt)



    



