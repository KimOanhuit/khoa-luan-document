import snap
import pandas as pd
import numpy as np
import random
import csv

g = snap.TNGraph.New()

def readWiki():
        f = open('Dataset/WikiDataset.txt', 'r')
        file1 = open('Dataset/WikiDataset.csv', 'a+')
        file1.write('SRC,TGT,Sign\n')
        find_user = dict()
        information = dict() #'SRC_ID TGT_ID':'VOT','RES','YEA','DAT','TXT'
        left_info = []
        right_info = []
        id = 0
        data = ''
        for line in f.readlines():
            if line == '\n':
                continue
            line = line.replace("\n","")
            part = line.split(":")
            
            if part[0] == "SRC":
                if part[1] not in find_user:
                    id += 1
                    find_user[part[1]] = id
                    g.AddNode(id)
                left_info.append(find_user[part[1]])
            
            elif part[0] == "TGT":
                if part[1] not in find_user:
                    id += 1
                    find_user[part[1]] = id
                    g.AddNode(id)
                left_info.append(find_user[part[1]])
                g.AddEdge(left_info[0], left_info[1])

            elif part[0] == "TXT":
                right_info.append(part[1])
                information["".join(str(left_info))] = right_info
                right_info = []
                left_info = []
            else:
                right_info.append(part[1])
        
        # for key in information.keys():
            # input = key.replace('[', '').replace(']', '').replace(',', '')
            # search = input.find(' ')
            # file1.write(input[:search] + ',')
            # file1.write(input[search:] + ',')
            # if information[key][0] == '-1':
            #     Sign = '-1'
            #     file1.write('-1\n')
            # if information[key][0] == '1':
            #     Sign = '1'
            #     file1.write('1\n')

        # count = 0
        # for key in information.keys():
        #     if information[key][0] == '-1':
        #         count += 1
        #         input = key.replace('[', '').replace(']', '').replace(',', '')
        #         search = input.find(' ')
        #         file1.write(input[:search] + ',')
        #         file1.write(input[search:] + ',-1\n')
        #         del information[key]

        # for i in range(1, count + 1):
        #     key = random.choice(information.keys())
        #     input = key.replace('[', '').replace(']', '').replace(',', '')
        #     search = input.find(' ')
        #     file1.write(input[:search] + ',')
        #     file1.write(input[search:] + ',1\n')
        #     del information[key]

        # file1.close()

def getMaxNode(Data):
    maxNode = 0
    for i in range(len(Data)):
        maxNode = max(maxNode, int(Data.at[i,'SRC']), int(Data.at[i,'TGT']))
    return maxNode

def AdjacencyMatrix(Data):
    maxNode = getMaxNode(Data)
    len_data = len(Data)
    MatrixData = [[0 for x in range(351)] for y in range(351)] 
    
    for i in range(1,len_data):
        SRC  = int(Data.at[i,'SRC'])
        TGT  = int(Data.at[i,'TGT'])
        Sign = int(Data.at[i,'Sign'])
        if SRC <= 350 and TGT <= 350:
            MatrixData[SRC][TGT] = Sign

    f_matrix = open('Dataset/Matrix.txt', 'a+')
    for i in range(1,351):
        for j in range(351):
            f_matrix.write(MatrixData[i][j])
            # f_matrix.write('\n')

    return MatrixData


# readWiki()
Data = pd.read_csv('Dataset/WikiDataset.csv')
AdjacencyMatrix(Data)
#getMaxNode(Data)