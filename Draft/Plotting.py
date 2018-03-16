import snap

G = snap.LoadEdgeList(snap.PNGraph, "GrapWiki.txt", 0, 1)
snap.PlotInDegDistr(G, "wikiInDeg", "wiki-RFA In Degree")