import snap

G = snap.LoadEdgeList(snap.PNGraph, "GrapWiki.txt", 0, 1)
graph = snap.TNGraph.New()


G1 = snap.GetBfsTree(G, 1, False, False)
G2 = snap.GetBfsTree(G, 2, False, False)
G3 = snap.GetBfsTree(G, 3, False, False)
G4 = snap.GetBfsTree(G, 4, False, False)
G5 = snap.GetBfsTree(G, 5, False, False)
G6 = snap.GetBfsTree(G, 6, False, False)
G7 = snap.GetBfsTree(G, 7, False, False)

G8 = snap.GetBfsTree(G, 1, True, False)
G9 = snap.GetBfsTree(G, 2, True, False)
G10 = snap.GetBfsTree(G, 3, True, False)
G11 = snap.GetBfsTree(G, 4, True, False)
G12 = snap.GetBfsTree(G, 5, True, False)
G13 = snap.GetBfsTree(G, 6, True, False)
G14 = snap.GetBfsTree(G, 7, True, False)

for EI in G1.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G2.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G3.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G4.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G5.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G6.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G7.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G8.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G9.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G10.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G11.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G12.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G13.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())
for EI in G14.Edges():
    print "Edge from %d to %d in generated tree." % (EI.GetSrcNId(), EI.GetDstNId())



# snap.SaveEdgeList(graph, "GraphWiki1.txt", "List of edges")

print "Graph: Nodes %d, Edges %d" % (G1.GetNodes(), G1.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G2.GetNodes(), G2.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G3.GetNodes(), G3.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G4.GetNodes(), G4.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G5.GetNodes(), G5.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G6.GetNodes(), G6.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G7.GetNodes(), G7.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G8.GetNodes(), G8.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G9.GetNodes(), G9.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G10.GetNodes(), G10.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G11.GetNodes(), G11.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G12.GetNodes(), G12.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G13.GetNodes(), G13.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G14.GetNodes(), G14.GetEdges())
print "Graph: Nodes %d, Edges %d" % (G.GetNodes(), G.GetEdges())