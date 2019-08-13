from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC(gamma='scale')
clf.fit(X, y)


def feature_encode(tweets, indexer):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    x = vectorizer.fit_transform(tweets).toarray().append(indexer.get(tweet))


def read_in(file_path):
    df = pd.read_csv(file_path, header=0, sep='\t', dtype={'id': str})
    ids = []
    tweets = []
    labels = []
    for index, row in df.iterrows():
        tweets.append(row.tweet)
        ids.append(row.ids)
        if row.subtask_a == "NOT":
            labels.append(0)
        else:
            labels.append(1)
    return ids, tweets, labels

