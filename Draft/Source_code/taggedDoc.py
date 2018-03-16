# @author: JapHien

from gensim import utils
from gensim.models.doc2vec import TaggedDocument
import random
# functions
from general_function import *

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split()+k_ngram(utils.to_unicode(line),2), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)