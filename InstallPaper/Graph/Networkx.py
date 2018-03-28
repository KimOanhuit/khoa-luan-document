import networkx as nx
import pandas as pd

class Wiki():

    # Data = pd.read_csv('Dataset/WikiDataset.csv')
    wikiGraph = None

    def __init__(self):
        
        self.wikiGraph = nx.DiGraph()

    def wikiNetworkx(self, Data):
        for i in range(1,len(Data)):
            SRC  = int(Data.at[i,'SRC'])
            self.wikiGraph.add_node(SRC)
            TGT  = int(Data.at[i,'TGT'])
            self.wikiGraph.add_node(TGT)
            Sign = int(Data.at[i,'Sign'])
            self.wikiGraph.add_edge(SRC, TGT, weight = Sign)

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
        print "IN: " + str(triadList)

    def computeFeatures(self,Data):
        f = open('../Dataset/Features.txt','w')
        # first = "SRC\tTGT\tSign\tIn+1\tIn-1\tOut+1\tOut-1\tIn+2\tIn-2\tOut+2\tOut-2\tCommonNeighbors\t"
        first = "SRC\tTGT\tSign\t"
        second = "FFpp\tFFpm\tFFmp\tFFmm\tFBpp\tFBpm\tFBmp\tFBmm\tBFpp\tBFpm\tBFmp\tBFmm\tBBpp\tBBpm\tBBmp\tBBmm\n"
        firstLine = first + second
        f.write(firstLine)
    
        edges = self.wikiGraph.edges
        unG = self.wikiGraph.to_undirected(reciprocal=False)
        for edge in edges:#pair, score in self.authorPairScore.iteritems():
            test1 = edge[0]
            test2 = edge[1]
            if test1 == test2:
                continue
            sign = self.wikiGraph.get_edge_data(test1,test2)
            f.write(str(test1) + "\t" + str(test2) + "\t" + str(sign['weight']) + "\t")

            neighbors1 = nx.all_neighbors(self.wikiGraph,test1)#self.graph.neighbors(test1)
            neighbors2 = nx.all_neighbors(self.wikiGraph,test2)#self.graph.neighbors(test2)

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
                    tmSign = self.wikiGraph.get_edge_data(test1,neighbor1,default={'weight':0})
                    tmpSign = tmSign['weight']
                    if tmpSign < 0:
                        minusCountOut += 1
                    elif tmpSign > 0:
                        plusCountOut += 1
                    else:
                        tmSign = self.wikiGraph.get_edge_data(neighbor1,test1,default={'weight':0})
                        tmpSign = tmSign['weight']
                        if tmpSign < 0:
                            minusCountIn += 1
                        elif tmpSign > 0:
                            plusCountIn += 1
            commonNeigh = sorted(nx.common_neighbors(unG,test1,test2))
            for inode in commonNeigh:
                self.computeTriads(inode,test1,test2,triadList,self.wikiGraph)
                comment = self.computeTriads(inode,test1,test2,triadList,self.wikiGraph)
                print comment

            commonNeighbors = len(commonNeigh)

            # text1 = str(plusCountIn) + "\t" + str(minusCountIn) + "\t" + str(plusCountOut) + "\t" + str(minusCountOut) + "\t"
            # text2 = str(rplusCountIn) + "\t" + str(rminusCountIn) + "\t" + str(rplusCountOut) + "\t" + str(rminusCountOut) + "\t"
            #final = text1 + text2 + str(commonNeighbors)
            final = str(commonNeighbors)
            for item in triadList:
                final += "\t" + str(item)

            f.write(final+"\n")
        f.close()


