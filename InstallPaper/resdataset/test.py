from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import random
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

iris = load_iris()
print iris
X, y = iris.data[:-2,:], iris.target[:-2]
print X
print str(len(X))
print str(len(y))
print y
logistic = LogisticRegression()
logistic.fit(X,y)
print str(logistic.predict(iris.data[-2:,:])) + str(iris.target[-2])
print str(logistic.predict_proba(iris.data[-2:,:]))