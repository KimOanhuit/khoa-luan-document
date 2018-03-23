import snap

class GraphCreator(object):
    graph = None
    graphBFS1 = None
    graphBFS2 = None
    graphBFS3 = None
    graphBFS4 = None
    graphBFS5 = None
    graphBFS6 = None
    graphBFS7 = None
    graphBFS8 = None
    graphBFS9 = None
    graphBFS10 = None

    def __init__(self):
        self.graph = snap.TNGraph.New()
        self.graphBFS1 = snap.TNGraph.New()
   
    def readWiki(self):
        f = open('Dataset/WikiDataset.txt', 'r')
        find_user = dict()
        information = dict() #'SRC_ID TGT_ID':'VOT','RES','YEA','DAT','TXT'
        left_info = []
        right_info = []
        id = 0

        for line in f.readlines():
            if line == '\n':
                continue
            line = line.replace("\n","")
            part = line.split(":")
            
            if part[0] == "SRC":
                if part[1] not in find_user:
                    id += 1
                    find_user[part[1]] = id
                    self.graph.AddNode(id)
                left_info.append(find_user[part[1]])
            
            elif part[0] == "TGT":
                if part[1] not in find_user:
                    id += 1
                    find_user[part[1]] = id
                    self.graph.AddNode(id)
                left_info.append(find_user[part[1]])
                self.graph.AddEdge(left_info[0], left_info[1])
                
            elif part[0] == "TXT":
                right_info.append(part[1])
                information["".join(str(left_info))] = right_info
                right_info = []
                left_info = []
            else:
                right_info.append(part[1])

        #snap.SaveEdgeList(self.graph, "Dataset/GraphWiki.txt", "List of edges")
        #print "graph: Nodes %d, Edges %d" % (self.graph.GetNodes(), self.graph.GetEdges())
        print information

    def randomNodes(self):
        listNodes = []
        for i in range(0, 10):
            NId = self.graph.GetRndNId()
            if NId not in listNodes:
                listNodes.append(NId)

        # file1 = open('Dataset/graph1.csv', 'a+')
        # file1.write('SRC,TGT,Sign\n')
        # data1 = ''
        # file2 = open('Dataset/graph2.csv', 'a+')
        # file2.write('SRC,TGT,Sign\n')
        # data2 = ''
        # file3 = open('Dataset/graph3.csv', 'a+')
        # file3.write('SRC,TGT,Sign\n')
        # data3 = ''
        # file4 = open('Dataset/graph4.csv', 'a+')
        # file4.write('SRC,TGT,Sign\n')
        # data4 = ''
        # file5 = open('Dataset/graph5.csv', 'a+')
        # file5.write('SRC,TGT,Sign\n')
        # data5 = ''
        # file6 = open('Dataset/graph6.csv', 'a+')
        # file6.write('SRC,TGT,Sign\n')
        # data6 = ''
        # file7 = open('Dataset/graph7.csv', 'a+')
        # file7.write('SRC,TGT,Sign\n')
        # data7 = ''
        # file8 = open('Dataset/graph8.csv', 'a+')
        # file8.write('SRC,TGT,Sign\n')
        # data8 = ''
        # file9 = open('Dataset/graph9.csv', 'a+')
        # file9.write('SRC,TGT,Sign\n')
        # data9 = ''
        # file10 = open('Dataset/graph10.csv', 'a+')
        # file10.write('SRC,TGT,Sign\n')
        # data10 = ''

        self.graphBFS1.AddNode(listNodes[0])
        queue = [listNodes[0]]
        dem = self.graphBFS1.GetNodes()
        while queue:
            node = queue.pop(0)
            for n in self.graph.Nodes():

                if dem < 350 and self.graphBFS1.n.GetId() not in self.graphBFS1:
                    self.graphBFS1.AddNode(self.graphBFS1.n.GetId())
        
        # print listNodes
        # return listNodes

    # def computeBFS(self):
    #     file1 = open
    #     for start in self.randomNodes():
    #         queue = [start]

    #         while queue:
    #             node = queue.pop(0)
    #             for n in self.readWiki():
    #                 if self.graph.start.IsNbrNId(n) == True:
    #                     if n not in self.graphBFS.Nodes():
    #                         if self.graphBFS.GetNodes() <= 350:
    #                             queue.append(n)
    #                             self.graphBFS.AddNode(n)
    #                             print self.graphBFS
                            



    



