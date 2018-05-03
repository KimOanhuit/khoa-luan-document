import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn import cross_validation
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve_neg
from scipy import interp

import matplotlib.pyplot as plt

class Text(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe

    # Loai bo dau cau ".", "," ...
    def clean_text(self, dataframe, col):
        return dataframe[col].fillna('').apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x.lower()))\
                    .apply(lambda x: re.sub('\s+', ' ', x).strip())

    # Loai bo Stop word trong tieng anh: nhung tu hay xuat hien nhung khong co
    # nhieu y nghia nhu: of, a, an, the...
    def remove_stopwords(self, tokenized_words):
        self.stop_words = stopwords.words('english')
        return [[w.lower() for w in sent
                if (w.lower() not in stop_words)]
                for sent in tokenized_words]

    # Dem so lan xuat hien cua pattern
    def count_pattern(self, dataframe, col, pattern):
        dataframe = dataframe.copy()
        return dataframe[col].str.count(pattern)

    def split_on_word(self, text):
        if type(text) is list:
            return [regexp_tokenize(sentence, pattern="\w+(?:[-']\w+)*") for sentence in text]
        else:
            return regexp_tokenize(text, pattern="\w+(?:[-']\w+)*")

    def flatten_words(self, list1d, get_unique=False):
        qa = [s.split() for s in list1d]
        if get_unique:
            return sorted(list(set([w for sent in qa for w in sent])))
        else:
            return [w for sent in qa for w in sent]
    
    def predict(self, dataframe):
        # Lam sach cot "TXT"
        text = self.clean_text(dataframe, 'TXT')

        # Add text vao list
        text_list = text.values.tolist()

        # Vector hoa
        vocab = self.flatten_words(text_list, get_unique=True)
        feature_extraction = TfidfVectorizer(analyzer='word', min_df=1, ngram_range=(1,1),stop_words='english', vocabulary=vocab, max_features=10000)

        num = int(round(len(dataframe["TXT"])*0.6))
        X = feature_extraction.fit_transform(dataframe["TXT"][:num].values)
        y = dataframe["Sign"][:num].values
        kf = KFold(n_splits = 10, shuffle = False, random_state = None)
        accs = []

        tprs = []
        aucs_roc = []
        mean_fpr = np.linspace(0,1,100)

        aucs_PR = []
        aucs_negPR = []
        
        for train, test in kf.split(X):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
    
            # clf = SVC(kernel='linear', C = 1).fit(X_train, y_train)
            # pred = clf.predict(X_test)
            logistic = LogisticRegression()
            clf = logistic.fit(X_train, y_train)
            pred = logistic.predict(X_test)
            
            acc = accuracy_score(pred, y_test)
            accs.append(acc)
            print "Accuracy: ", round(acc,2)
            
            fpr, tpr, _ = roc_curve(y_test, pred)
            tprs.append(interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs_roc.append(roc_auc)
            print "AUC/ROC: ", round(roc_auc,2)

            precision, recall, thresholds = precision_recall_curve(y_test, pred)
            auc_PR = auc(recall, precision)
            aucs_PR.append(auc_PR)
            print "AUC/PR: ", round(auc_PR,2)

            precision_neg, recall_neg, thresholds_neg = precision_recall_curve_neg(y_test, pred)
            auc_negPR = auc(recall_neg, precision_neg)
            aucs_negPR.append(auc_negPR)
            print "AUC/negPR: ", round(auc_negPR,2)
        
        accs_sum = sum(accs)
        accs_length = len(accs)
        accs_mean = accs_sum / accs_length
        print "Accuracy Average: ", round(accs_mean,2)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        print "AUC/ROC Average: ", round(mean_auc,2)

        aucs_PR_sum = sum(aucs_PR)
        aucs_PR_length = len(aucs_PR)
        aucs_PR_mean = aucs_PR_sum / aucs_PR_length
        print "AUC/PR Average: ", round(aucs_PR_mean,2)

        aucs_negPR_sum = sum(aucs_negPR)
        aucs_negPR_length = len(aucs_negPR)
        aucs_negPR_mean = aucs_negPR_sum / aucs_negPR_length
        print "AUC/negPR Average: ", round(aucs_negPR_mean,2)















