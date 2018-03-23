import snap

def randomNodes():
    G = snap.LoadEdgeList(snap.PNGraph, 'Dataset/GraphWiki.txt', 0, 1)
    G1 = snap.TNGraph.New()
    listNodes = []
    for i in range(0, 10):
        NId = G.GetRndNId()
        if NId not in listNodes:
            listNodes.append(NId)

    G1.AddNode(listNodes[0])
    queue = [listNodes[0]]
    dem = 0
    l = []
    while queue:
        node = queue.pop(0)
        for n in G.Nodes():
            if dem < 349 and n.GetId() not in l:
                dem += 1
                l.append(n.GetId())
                G1.AddNode(n.GetId())
                G1.AddEgde(n.GetId())
    
    snap.SaveEdgeList(G1, "Dataset/Graph1.txt", "List of edges")
    print "graph: Nodes %d, Edges %d" % (G1.GetNodes(), G1.GetEdges())
    

randomNodes()       