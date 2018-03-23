import sys
import snap

information = dict() #'SRC_ID TGT_ID':'VOT','RES','YEA','DAT','TXT'
left_info = []
right_info = []
find_user = dict() # user_name : user_id
id = 0
graph = snap.TNGraph.New()
graph = snap.TNGraph.New()
graph1 = snap.TNGraph.New()
graph2 = snap.TNGraph.New()
graph3 = snap.TNGraph.New()
graph4 = snap.TNGraph.New()
graph5 = snap.TNGraph.New()
graph6 = snap.TNGraph.New()
graph7 = snap.TNGraph.New()
graph8 = snap.TNGraph.New()
graph9 = snap.TNGraph.New()
graph10 = snap.TNGraph.New()

i = 0

for line in sys.stdin.readlines():
    if line == '\n':
        i+=1
        continue
    line = line.replace("\n","")
    part = line.split(":")
    if part[0] == "SRC":
        if part[1] not in find_user:
            id += 1
            find_user[part[1]] = id
            graph.AddNode(id)
            graph1.AddNode(id)
            graph2.AddNode(id)
            graph3.AddNode(id)
            graph4.AddNode(id)
            graph5.AddNode(id)
            graph6.AddNode(id)
            graph7.AddNode(id)
            graph8.AddNode(id)
            graph9.AddNode(id)
            graph10.AddNode(id)
            
        left_info.append(find_user[part[1]])
    elif part[0] == "TGT":
        if part[1] not in find_user:
            id += 1
            find_user[part[1]] = id
            graph.AddNode(id)
            graph1.AddNode(id)
            graph2.AddNode(id)
            graph3.AddNode(id)
            graph4.AddNode(id)
            graph5.AddNode(id)
            graph6.AddNode(id)
            graph7.AddNode(id)
            graph8.AddNode(id)
            graph9.AddNode(id)
            graph10.AddNode(id)
        left_info.append(find_user[part[1]])
        graph.AddEdge(left_info[0], left_info[1])
        if i % 10 == 0:
            graph10.AddEdge(left_info[0], left_info[1])                
        elif i % 10 == 1:
            graph1.AddEdge(left_info[0], left_info[1])                                
        elif i % 10 == 2:
            graph2.AddEdge(left_info[0], left_info[1])                                
        elif i % 10 == 3:
            graph3.AddEdge(left_info[0], left_info[1])                                
        elif i % 10 == 4:
            graph4.AddEdge(left_info[0], left_info[1])                                
        elif i % 10 == 5:
            graph5.AddEdge(left_info[0], left_info[1])                                
        elif i % 10 == 6:
            graph6.AddEdge(left_info[0], left_info[1])                                
        elif i % 10 == 7:
            graph7.AddEdge(left_info[0], left_info[1])                                
        elif i % 10 == 8:
            graph8.AddEdge(left_info[0], left_info[1])                                
        else:
            graph9.AddEdge(left_info[0], left_info[1])                                

    elif part[0] == "TXT":
        right_info.append(part[1])
        information[" ".join(str(left_info))] = right_info
        right_info = []
        left_info = []
    else:
        right_info.append(part[1])

# snap.SaveEdgeList(graph, "GrapWiki.txt", "List of edges")
snap.SaveEdgeList(graph1, "GrapWiki1.txt", "List of edges")
snap.SaveEdgeList(graph2, "GrapWiki2.txt", "List of edges")
snap.SaveEdgeList(graph3, "GrapWiki3.txt", "List of edges")
snap.SaveEdgeList(graph4, "GrapWiki4.txt", "List of edges")
snap.SaveEdgeList(graph5, "GrapWiki5.txt", "List of edges")
snap.SaveEdgeList(graph6, "GrapWiki6.txt", "List of edges")
snap.SaveEdgeList(graph7, "GrapWiki7.txt", "List of edges")
snap.SaveEdgeList(graph8, "GrapWiki8.txt", "List of edges")
snap.SaveEdgeList(graph9, "GrapWiki9.txt", "List of edges")
snap.SaveEdgeList(graph10, "GrapWiki10.txt", "List of edges")

print "Graph: Nodes %d, Edges %d" % (graph.GetNodes(), graph.GetEdges())
print "Graph1: Nodes %d, Edges %d" % (graph1.GetNodes(), graph1.GetEdges())
print "Graph2: Nodes %d, Edges %d" % (graph2.GetNodes(), graph2.GetEdges())
print "Graph3: Nodes %d, Edges %d" % (graph3.GetNodes(), graph3.GetEdges())
print "Graph4: Nodes %d, Edges %d" % (graph4.GetNodes(), graph4.GetEdges())
print "Graph5: Nodes %d, Edges %d" % (graph5.GetNodes(), graph5.GetEdges())
print "Graph6: Nodes %d, Edges %d" % (graph6.GetNodes(), graph6.GetEdges())
print "Graph7: Nodes %d, Edges %d" % (graph7.GetNodes(), graph7.GetEdges())
print "Graph8: Nodes %d, Edges %d" % (graph8.GetNodes(), graph8.GetEdges())
print "Graph9: Nodes %d, Edges %d" % (graph9.GetNodes(), graph9.GetEdges())
print "Graph10: Nodes %d, Edges %d" % (graph10.GetNodes(), graph10.GetEdges())
