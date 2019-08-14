from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tweetinvertedindex import Index, createTweetsInvertedIndex
from twokenize import tokenize
from nltk.stem.snowball import EnglishStemmer
import nltk
import numpy as np
from sklearn import metrics


def feature_encode(inverted_indexer, tr_ids, tr_tweets, ts_ids, ts_tweets):
    """
    encode test and train set at once.
    :param inverted_indexer:
    :param tr_ids: training...
    :param tr_tweets: training...
    :param ts_ids: testing...
    :param ts_tweets: testing...
    :return: encoded training, testing set
    """
    print("Encoding features...")
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)
    words_weights = inverted_indexer.getPolarizedWord()
    tr_x = vectorizer.fit_transform(tr_tweets).toarray()
    tr_x = np.append(tr_x, inverted_indexer.get_tweets_weights_by_ids(words_weights, tr_ids), axis=1)

    ts_x = vectorizer.transform(ts_tweets).toarray()
    ts_x = np.append(ts_x, inverted_indexer.get_test_weights(words_weights, ts_tweets, ts_ids), axis=1)
    return tr_x, ts_x


def read_in(file_path='./olid-training-v1.0.tsv', _type='train'):
    print("Loading in %sing set..." % _type)
    if _type == 'test':
        file_path = './test.data'
    df = pd.read_csv(file_path, header=0, sep='\t', dtype={'id': str})
    print(df.groupby('subtask_a').count())
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
    return ids, tweets, labels


def run_svm(x_train, y, x_test):
    print("Running SVM...\n")
    classifier = svm.SVC(gamma='scale')
    classifier.fit(x_train, y)
    return classifier.predict(x_test)


if __name__ == '__main__':
    ids_train, tweets_train, labels_train = read_in(_type='train')
    ids_test, tweets_test, labels_test = read_in(_type='test')
    indexer = createTweetsInvertedIndex(Index(tokenizer=tokenize,
                                              stemmer=None,
                                              stopwords=nltk.corpus.stopwords.words('english')))
    X_train, X_test = feature_encode(indexer, ids_train, tweets_train, ids_test, tweets_test)
    pred = run_svm(X_train, labels_train, X_test)
    print(metrics.classification_report(labels_test, pred))
