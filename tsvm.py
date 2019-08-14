from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tweetinvertedindex import Index, get_tweets_weights_by_ids, get_test_weights, createTweetsInvertedIndex
from twokenize import tokenize
from nltk.stem.snowball import EnglishStemmer
import nltk
from separateOffLang import make_test
import numpy as np
from sklearn import metrics


def feature_encode(inverted_indexer, tr_ids, tr_tweets, ts_ids, ts_tweets):
    """
    encode test and train set at once.
    :param inverted_indexer:
    :param tr_ids:
    :param tr_tweets:
    :param ts_ids:
    :param ts_tweets:
    :return: encoded training, testing set
    """
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=200)
    words_weights = inverted_indexer.getPolarizedWord()
    tr_x = vectorizer.fit_transform(tr_tweets).toarray()
    tr_x = np.append(tr_x, get_tweets_weights_by_ids(inverted_indexer, words_weights, tr_ids), axis=1)

    ts_x = vectorizer.transform(ts_tweets).toarray()
    ts_x = np.append(ts_x, get_test_weights(words_weights, ts_tweets, ts_ids), axis=1)
    return tr_x, ts_x


def read_in(file_path='./olid-training-v1.0.tsv', _type='train'):
    if _type == 'train':
        print("Loading in training set...")
        df = pd.read_csv(file_path, header=0, sep='\t', dtype={'id': str})
    if _type == 'test':
        print("Loading in testing set...")
        df = file_path
    ids = []
    tweets = []
    labels = []
    for index, row in df.iterrows():
        tweets.append(row.tweet)
        ids.append(row.id)
        if row.subtask_a == "NOT":
            labels.append(0)
        else:
            labels.append(1)
    print("Loading %s data finished!" % _type)
    return ids, tweets, labels


def run_svm(x_train, y, x_test):
    classifier = svm.SVC(gamma='scale')
    classifier.fit(x_train, y)
    return classifier.predict(x_test)


if __name__ == '__main__':
    tr_ids, tr_tweets, labels_train = read_in(_type='train')
    ts_ids, ts_tweets, labels_test = read_in(make_test(), _type='test')
    indexer = createTweetsInvertedIndex(Index(tokenize,
                                              EnglishStemmer(),
                                              nltk.corpus.stopwords.words('english')))
    X_train, X_test = feature_encode(indexer, tr_ids, tr_tweets, ts_ids, ts_tweets)
    pred = run_svm(X_train, labels_train, X_test)
    print(metrics.classification_report(labels_test, pred))
