import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
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
        
        # # PCA reduction to 2 dimension
        # svd = TruncatedSVD(n_components=2)
        # SVD = svd.fit_transform(X)
        # dataframe['firstSVD']
        # principalDf = pd.DataFrame(data = SVD, columns = ['firstSVD', 'secondSVD'])
        # finalDf = pd.concat([principalDf,dataframe], axis = 1)
        # finalDf.to_csv('Dataset/Wiki/test.csv')

        # principalDf = pd.DataFrame(data = SVD, columns = ['firstSVD', 'secondSVD'])
        # finalDf = pd.concat([principalDf,dataframe], axis = 1)
        # finalDf.to_csv('Dataset/Wiki/testWithText.csv')

        #KFolds

        num = int(round(len(dataframe["TXT"])*0.15))
        X = feature_extraction.fit_transform(dataframe["TXT"].values)
        y = dataframe["Sign"].values

        kf = KFold(n_splits = 10, shuffle = False, random_state = None)

        for train, test in kf.split(X):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
        
            logistic = LogisticRegression()
            clf = logistic.fit(X_train, y_train)
            pred = logistic.predict(X_test)
                
            acc = accuracy_score(pred, y_test)
            print "Accuracy: ", round(acc,3)
                
            fpr, tpr, _ = roc_curve(y_test, pred)
            roc_auc = auc(fpr, tpr)
            print "AUC/ROC: ", round(roc_auc,3)




















