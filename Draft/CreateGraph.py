import sys
import snap

information = dict() #'SRC_ID TGT_ID':'VOT','RES','YEA','DAT','TXT'
left_info = []
right_info = []
find_user = dict() # user_name : user_id
id = 0
graph = snap.TNGraph.New()

for line in sys.stdin.readlines():
    if line == '\n':
        continue
    line = line.replace("\n","")
    part = line.split(":")
    if part[0] == "SRC":
        if part[1] not in find_user:
            id += 1
            find_user[part[1]] = id
            graph.AddNode(id)
        left_info.append(find_user[part[1]])
    elif part[0] == "TGT":
        if part[1] not in find_user:
            id += 1
            find_user[part[1]] = id
            graph.AddNode(id)
        left_info.append(find_user[part[1]])
        graph.AddEdge(left_info[0], left_info[1])
    elif part[0] == "TXT":
        right_info.append(part[1])
        information[" ".join(str(left_info))] = right_info
        right_info = []
        left_info = []
    else:
        right_info.append(part[1])

snap.SaveEdgeList(graph, "GraphWiki.txt", "List of edges")
print "Graph: Nodes %d, Edges %d" % (graph.GetNodes(), graph.GetEdges())
