import re
import networkx as nx
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from scipy.sparse import hstack
import matplotlib.pyplot as plt

def readWiki():
    f = open('Dataset/Wiki/WikiDataset.txt', 'r')
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
            left_info.append(find_user[part[1]])
            
        elif part[0] == "TGT":
            if part[1] not in find_user:
                id += 1
                find_user[part[1]] = id
            left_info.append(find_user[part[1]])
                
        elif part[0] == "TXT":
            right_info.append(part[1])
            information["".join(str(left_info))] = right_info
            right_info = []
            left_info = []
        else:
            right_info.append(part[1])

    f = open ('Dataset/Wiki/DatasetFull.csv', 'a+')
    f.write('SRC,TGT,Sign,TXT\n')
    for key in information.keys():
        input = key.replace('[', '').replace(']', '').replace(',', '')
        find = input.find(' ')
        SRC = input[:find]
        TGT = input[find:]
        TGT = TGT.replace(' ','')
        Sign = information[key][0]
        TXT = information[key][4].replace("'","").replace(",", "").replace(".", "").replace(":", "").replace(";", "").replace(":", "").replace("--", "")
        data = SRC + ',' + TGT + ',' + Sign + ',' + TXT
        f.write(data + '\n')
    f.close()

def resdataset():

    Data = open('Dataset/Wiki/DatasetFullTestCopy.csv', 'w')
    f = open('Dataset/Wiki/DatasetFullTest.csv', 'r')
    for line in f.readlines():
        
        # line = re.sub(r'<.*?$',' ', line)

        # if '1,[[File\n' in line:
        #    continue
        # if '1,(\n' in line:
        #    continue
        # if '1,+\n' in line:
        #    continue
        if '1,\n' in line:
           continue

        # if '1,--\n' in line:
        #    continue
        
        # if '1,\n' in line:
        #    line = line.replace('1,\n', 'Yes')
        #    count += 1
        # if '-1,\n' in line:
        #    line = line.replace('1,\n', 'No')
        #    count += 1
        # if '1,[[Image\n' in line:
        #     continue
        # if '1,[http\n' in line:
        #     continue
        Data.write(line)
    Data.close()

    

    # Data = open('Dataset/Wiki/test.csv', 'a+')
    # f = open('Dataset/Wiki/DatasetFullCopy.csv', 'r')
    
    # for line in f.readlines():
        
    #     if '1,&mdash\n' in line:
    #         line =line.replace('1,&mdash\n', '1,Yes\n')
    #     if '-1,\n' in line:
    #         line = line.replace('-1,\n','-1,No\n')
    #     if '1,\n' in line:
    #         line = line.replace('1,\n','1,Yes\n')
        # if '-1,[[Image\n' in line:
        #     line = line.replace('-1,[[Image\n', '-1,No\n')
        # if '1,[[Image\n' in line:
        #     line = line.replace('1,[[Image\n', '1,Yes\n')
        # if '-1,[[WP\n' in line:
        #     line = line.replace('-1,[[WP\n', '-1,No\n')
        # if '1,[[WP\n' in line:
        #     line = line.replace('1,[[WP\n', '1,Yes\n')
        # if '-1,[[WP\n' in line:
        #     line = line.replace('-1,[[WP\n', '-1,No\n')
        # if '1,[[WP\n' in line:
        #     line = line.replace('1,[[WP\n', '1,Yes\n')
        # if '-1,[[User\n' in line:
        #     line = line.replace('-1,[[User\n', '-1,No\n')
        # if '1,[[User\n' in line:
        #     line = line.replace('1,[[User\n', '1,Yes\n')

def examine():
    f = open('Dataset/Wiki/FeaturesFullTest.txt','r')
    count = 0
    for line in f.readlines():
        if line == '\n':
            count += 1
    print count

def clean_text(dataframe, col):
    return dataframe[col].fillna('').apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                .apply(lambda x: re.sub('\s+', ' ', x).strip())


data = pd.read_csv('Dataset/Wiki/DatasetFullCopy.csv')
data['TXT'] = clean_text(data, 'TXT')
# data.to_csv('Dataset/Wiki/DatasetFullTest.csv')
# finalDf = pd.concat([training['TXT_clean'],dataframe], axis = 1)

    
resdataset()


    

    



