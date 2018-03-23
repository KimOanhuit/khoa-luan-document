# -*- coding: utf8 -*-
# @author: JapHien
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# functions
from general_function import *

import pandas as pd
# random
import random
# numpy
import numpy
# classifier
from sklearn.naive_bayes import GaussianNB
# evaluate
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# etc
import logging
import sys, os, re, time
# visualize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=5)

start_time = time.time()
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

crawled_files = 'DATA/MATERIAL/'
save_folder = 'TRAINED_MODELS/'

# os.chdir(crawled_files)


models=list_models(save_folder)
for model in models:
    print (str(model) + ' -*- ' + models[model])

model_name = models[int(input('Enter model number: '))]
log.info('Loading model . . .')
try:
    model = Doc2Vec.load(save_folder + model_name)
except:
    model = Doc2Vec.load(save_folder + 'default_model.d2v')
#==============================================================================

sources = get_folder_name(crawled_files)
print (sources)
total_file = 0

#**********************************************************

prefix = get_prefix(sources)

total_file = count_files(crawled_files)
dimention = int(re.search('_(\d*)d', model_name).group(1))
num_topic = len(prefix)

#**********************************************************

# prepair data and label for training
data = numpy.zeros((total_file, dimention))
labels = numpy.zeros((total_file))

names = get_topic_names(prefix)
idx = 0
j = 0

for i in range (num_topic):
    while True:
        try:
            data[idx] = model.docvecs[prefix[i] + str(j)]
            labels[idx] = i
        except:
            j = 0
            break        
        j += 1
        idx += 1

#==============================================================================

log.info('Evaluating Classifier. . .')
k = int(input('\nK fold = '))
k_range = range (2, k)
k_scores = []

plt.rc('lines',linewidth=2, color='r')

for k in k_range:
    classifier = GaussianNB()
    scores = cross_val_score(classifier, data, labels, cv=k, n_jobs=-1)
    k_scores.append(scores.mean()*100)
    predicted = cross_val_predict(classifier, data, labels, cv=k)
    print(classification_report(labels, predicted,target_names=names))
#Compute confusion matrix
cnf_matrix = confusion_matrix(labels, predicted)

numpy.set_printoptions(precision=5)

# Plot non-normalized confusion matrix
fig = plt.figure(figsize = (30,20))
plot_confusion_matrix(cnf_matrix, classes=names, title='Confusion matrix, without normalization')
plt.plot()
try:
    plt.show()
except:
    pass

print ('Cross validated in', k_range)
print('Max Accuracy: ', max(k_scores))
print('Average Accuracy: ', numpy.mean(k_scores))


plt.plot(k_range, k_scores)
plt.xlabel('Value of K-fold')
plt.ylabel('Cross validated accuracy')
plt.show()
print(("--- %s mins ---" % ((time.time() - start_time)/60)))
