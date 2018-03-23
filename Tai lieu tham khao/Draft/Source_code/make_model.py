# -*- coding: utf8 -*-
# @author: JapHien

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# functions
from general_function import *

# random
import random
# numpy
import numpy
# taggedDoc
import taggedDoc as TD
# evaluate
from sklearn.utils import shuffle

# etc
import logging
import sys, os, re, time
# visualize
import matplotlib.pyplot as plt

start_time = time.time()

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

crawled_files = 'DATA/MATERIAL/'
cooked_files = 'DATA/COOKED/'
save_folder = 'TRAINED_MODELS/'

cooked = int(input('Did you prepaire your meal? (1/0):'))
if cooked:
	print ('Okay! have a good meal!')
else:
	print ('Cooking . . .')
	cook_files(crawled_files, cooked_files)
	print ('Enjoy yourself :)!')

os.chdir(cooked_files)
# =============================================================================

log.info('source load')
sources = get_sources(cooked_files)
print ('Your menu:')
for idx in sources:
	print (sources[idx])


#******************************************************************************

min_count = int(input('\nMincount = '))
window = int(input('Window = '))
size = int(input('Size = '))
epochs = int(input('Epochs = '))
print ('\n')
save_name = 'JAP_'+str(min_count)+'c_'+str(window)+'w_'+str(size)+'d_'+str(epochs)+'e'+'.d2v'

#==============================================================================
log.info('Tagging Document . . .')
sentences = TD.TaggedLineSentence(sources)

log.info('Building vocabulary . . .')
model = Doc2Vec(min_count=min_count, window=window, size=size, workers=10)
model.build_vocab(sentences.to_array())

# training
log.info('Epoch')
for epoch in range(epochs):
    log.info('EPOCH: {}'.format(epoch))
    model.train(sentences.sentences_perm(),total_examples=model.corpus_count,epochs=model.iter)

log.info('Saving model . . .')
os.chdir('../../')
try:
    model.save(save_folder + save_name)
except:
    model.save(save_folder + 'default_model.d2v')
