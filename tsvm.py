from sklearn import svm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tweetinvertedindex import Index, create_tweets_inverted_index
from twokenize import tokenize
from nltk.stem.snowball import EnglishStemmer
import nltk
import numpy as np
from sklearn import metrics
from termcolor import colored
from fromGATE import GateExporter
import datetime
import os
from DataProvider import TweetWarehouse

tweet_warehouse = TweetWarehouse()


def encode_feature(tr_ids, tr_tweets, ts_ids, ts_tweets, inverted_indexer, gate_exporter):
    """
    encode test and train set at once. NOTE: ids and tweets must be inter-correspond
    :param inverted_indexer: optional, if want feature from iindexer
    :param tr_ids: training...
    :param tr_tweets: training...
    :param ts_ids: testing...
    :param ts_tweets: testing...
    :param gate_exporter optional, if want feature from GATE
    :return: encoded training, testing set
    """
    print("Encoding features...")
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=32)
    tr_x = vectorizer.fit_transform(tr_tweets).toarray()
    ts_x = vectorizer.transform(ts_tweets).toarray()
    if inverted_indexer:
        tr_x = np.append(tr_x, inverted_indexer.get_tweets_weights_by_ids(tr_ids), axis=1)
        ts_x = np.append(ts_x, inverted_indexer.get_test_weights(ts_tweets, ts_ids), axis=1)
    if gate_exporter:
        tr_x = np.append(tr_x, gate_exporter.get_length_feature(tr_ids), axis=1)
        ts_x = np.append(ts_x, gate_exporter.get_length_feature(ts_ids), axis=1)
    return tr_x, ts_x


def read_in(_type='train', warehouse=None):
    """
    read in data and init TweetWarehouse
    :param _type:
    :param warehouse: where data store in
    :return: trainable format
    """
    print("Loading in %sing set..." % _type)
    file_path = './olid-training-v1.0.tsv'
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
            warehouse.put_in(row.id, row.tweet, 0)
        else:
            labels.append(1)
            warehouse.put_in(row.id, row.tweet, 1)
    return ids, tweets, labels


def run_svm(x_train, y, x_test):
    print("Running SVM...\n")
    classifier = svm.SVC(gamma=0.01, class_weight={0: 1, 1: 2}, C=1.5)
    classifier.fit(x_train, y)
    return classifier.predict(x_test)


def compare(main_log, logs, verbose=True):
    main_log = open(main_log, 'r')
    main_records = set([l.strip() for l in main_log.readlines()])
    main_log.close()
    for log in logs:
        f_log = open('./log/' + log, 'r')
        records = set([l.strip() for l in f_log.readlines()])
        f_log.close()
        print("Comparing with %s:" % log)
        inter = main_records.intersection(records)
        gains = main_records.difference(inter)
        losts = records.difference(inter)
        if verbose:
            for gain in gains:
                print(colored("Gain id: %s\tlabel: %s\t%s" % (gain,
                                                              tweet_warehouse.get_tweet(gain),
                                                              tweet_warehouse.get_label(gain)), 'green'))
            for lost in losts:
                print(colored("Gain id: %s\tlabel: %s\t%s" % (lost,
                                                              tweet_warehouse.get_tweet(lost),
                                                              tweet_warehouse.get_label(lost)), 'magenta'))
        else:
            print(colored("Gain: %s" % gains, 'green'))
            print(colored("Lost: %s" % losts, 'magenta'))


def error_analyse(ids, tweets, predictions, golds, mute_correct=True, verbose=True):
    print("Analysing errors...")
    previous_logs = os.listdir("./log")
    log_name = './log/%s.log' % datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    this_log = open(log_name, 'w')
    final = list(zip(predictions, golds))
    tp = [bundle[1] == 1 and bundle[0] == 1 for bundle in final]
    tn = [bundle[1] == 0 and bundle[0] == 0 for bundle in final]
    fp = [bundle[1] == 0 and bundle[0] == 1 for bundle in final]
    fn = [bundle[1] == 1 and bundle[0] == 0 for bundle in final]
    tp_tn = np.logical_or(tp, tn)
    print("fp=%d\tfn=%d\ttp=%d\ttn=%d" % (sum(fp), sum(fn), sum(tp), sum(tn)))
    for i in range(len(tp_tn)):
        if not tp_tn[i]:
            if golds[i] == 0:  # false positive
                print(colored("id: %s\tlabel: %s\t%s" % (ids[i], "NOT", tweets[i]), "yellow"))
            if golds[i] == 1:  # false negative
                print(colored("id: %s\tlabel: %s\t%s" % (ids[i], "OFF", tweets[i]), "red"))
        else:
            if not mute_correct:
                print(colored("%s" % tweets[i], "green"))
            this_log.write(ids[i]+'\n')
    this_log.close()
    if previous_logs:
        compare(log_name, previous_logs, verbose=verbose)


if __name__ == '__main__':
    ids_train, tweets_train, labels_train = read_in(_type='train', warehouse=tweet_warehouse)
    ids_test, tweets_test, labels_test = read_in(_type='test', warehouse=tweet_warehouse)
    indexer = create_tweets_inverted_index(Index(tokenizer=tokenize,
                                                 stemmer=None,
                                                 stopwords=nltk.corpus.stopwords.words('english')),
                                           zip(ids_train, tweets_train, labels_train))
    gate = GateExporter(tweets_train+tweets_test, ids_train+ids_test)
    X_train, X_test = encode_feature(ids_train, tweets_train, ids_test, tweets_test,
                                     inverted_indexer=indexer,
                                     gate_exporter=gate)
    pred = run_svm(X_train, labels_train, X_test)
    print(metrics.classification_report(labels_test, pred))
    error_analyse(ids_test, tweets_test, pred, labels_test, mute_correct=True, verbose=True)
