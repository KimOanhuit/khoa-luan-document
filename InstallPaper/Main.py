from GraphCreator import GraphCreator

def printMenu():
    print "-------------------------------------"
    print "               Menu              "
    print "-------------------------------------"
    print "Choose:"
    print "1. Create Wikipedia Graph full"
    print "2. Create Wikipedia Subgraph with BFS"
    print "3. Exact Triads Balance Featers"
    print "4. Training"
    print "5. Exit"

    try:
        reply = int(raw_input('Answer:'))
    except ValueError:
        print "Not a number"

    return reply

def main():
    while True:
        reply = printMenu()

        if reply == 1:
            gCreator = GraphCreator()
            gCreator.readWiki()
            # gCreator.randomNodes()
            # gCreator.computeBFS()

        elif reply == 5:
            break

if __name__ == '__main__':     # if the function is the main function ...
    main()

