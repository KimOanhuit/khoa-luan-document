# -*- coding: utf8 -*-
# @author: JapHien

import os, re
from nltk import ngrams
import matplotlib.pyplot as plt
import numpy
import itertools
from sklearn.metrics import confusion_matrix

def get_sources(folder_path):
    sources = {}
    for file_name in os.listdir(os.getcwd()):
        sources[file_name] = re.sub('\.txt', '', file_name)
    return sources

# n-gram
def k_ngram(string, n=1):
    gram_str = list(ngrams(string.split(), n))
    return [ ' '.join(gram) for gram in gram_str ]

def get_folder_name(path):
    return os.listdir(path)

def count_files(path):
    c = 0
    for f in os.listdir(path):
        c += len(os.listdir(path + '/' + f))
    return c

def count_file_in_folder(path):
    return len(os.listdir(path))

def get_folder_name(path):
    return os.listdir(path)

def get_prefix(sources):
    return sorted([re.sub('\.txt', '', file_name) + '_' for file_name in sources])

def get_topic_names(prefix):
	return [i.strip('_') for i in prefix]

def list_models(path):
    models = {}
    index = 1
    for model in os.listdir(path):
        if model.endswith('.d2v'):
            models[index] = model
            index += 1
    return models

def get_stopwords():
    tmp_stop_words = []
    stop_words = []
    for file in os.listdir('./stopword/'):
        print ('Found stop word file', file)
        with open ('./stopword/' + file) as f:
            for line in f:
                line = line.strip().replace('_', ' ')
                if line not in tmp_stop_words:
                    tmp_stop_words.append(line)
        for i in reversed(tmp_stop_words):
            stop_words.append(i)
    return stop_words

def cooking(file, stop_words):
    ls = []
    with open(file) as f:
        for string in f:
            if string != '':
                tmp = re.sub('[0-9;:,\.\?!%&\*\$\>\<\(\)\'\'\“\”\"\"/…\-\+\’]', '', string)
                tmp = re.sub('[\-\–\+\=\≥\\\]', '', tmp).strip().lower()
                tmp = re.sub('\s+', ' ', tmp)
                if tmp != '':
                    ls.append(tmp.strip())
        meal = ' '.join(ls)
        meaningful_words = [w for w in meal.split() if not w in stop_words]
        meal = ' '.join(meaningful_words)
    return (meal)

def cook_files(material, table):
    stop_words = get_stopwords()
    for stuff in os.listdir(material):
        print ('Working on', stuff)
        disk = open(table + re.sub('\.done', '', stuff) + '.txt', 'w')
        for item in os.listdir(material + stuff):
            content = ''
            raw =  material + stuff + os.sep + item
            meal = cooking(raw, stop_words)
            disk.write('\n'+meal)
        disk.close()

# plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')