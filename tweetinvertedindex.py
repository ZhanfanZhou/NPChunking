import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import pandas as pd
import math
from termcolor import colored
from twokenize import tokenize


class Index:

    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.__unique_id = 'UNKNOWN'
        self.info = {}
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    def lookup(self, word, stem=True):
        """
        Lookup a word in the index
        :return content of doc and its id
        """
        word = word.lower()
        if stem and self.stemmer:
            word = self.stemmer.stem(word)

        return [(self.documents.get(id, None), id) for id in self.index.get(word)]

    def add(self, document, doc_id, label):
        """
        Add a document string to the index
        """
        self.__unique_id = doc_id
        for token in [t.lower() for t in self.tokenizer(document)]:
            if token in self.stopwords:
                continue

            if self.stemmer:
                token = self.stemmer.stem(token)

            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)

        self.documents[self.__unique_id] = document
        self.info[self.__unique_id] = label

#       show tweet unique id and contents as well

    def getPolarizedWord(self):
        iidx = self.index
        label_info = self.info

        def getPorprotion(ids):
            off = 0
            for id in ids:
                if label_info[id] == 'OFF':
                    off += 1
            return off / float(len(ids)), len(ids)

        res = sorted([(word, getPorprotion(docs)) for word, docs in iidx.items()],
                     key=lambda x: -math.log(x[1][1], 2) * (pow((x[1][0] - 0.5), 2)))
        for r in res[:100]:
            print(r)
        return res

    def getExceptionals(self, p_words, n=200):
        for word, rate in p_words[0:n]:
            print(colored("looking up: %s, offensive rate = %f, count = %d" % (word, rate[0], rate[1]), "green"))
            docs = self.lookup(word, False)
            if rate == 1.0 or rate == 0.0:
                continue
            if rate[0] >= 0.5:
                for doc in docs:
                    if self.info.get(doc[1]) == "NOT":
                        print(doc[0])
            else:
                for doc in docs:
                    if self.info.get(doc[1]) == "OFF":
                        print(doc[0])


def createTweetsInvertedIndex(inverted_indexer, tweet_path='./olid-training-v1.0.tsv'):
    df = pd.read_csv(tweet_path, header=0, sep='\t', dtype={'id': str})
    for index, row in df.iterrows():
        inverted_indexer.add(row.tweet, row.id, row.subtask_a)
    return inverted_indexer


def get_tweets_weights_by_ids(inverted_indexer, words_weights):
    """
    :param inverted_indexer: indexer
    :param weights: from getPolarizedWord()
    :return:
    """
    ranks = {}
    for word, weight in words_weights:
        for id in inverted_indexer.index.get(word):
            ranks[id] = ranks.get(id, 0) + weight[0]
    return ranks


if __name__ == '__main__':
    indexer = createTweetsInvertedIndex(Index(tokenize,
                                              EnglishStemmer(),
                                              nltk.corpus.stopwords.words('english')))
    words_weights = indexer.getPolarizedWord()
    indexer.getExceptionals(words_weights)
    # print(len(indexer.lookup('fuck')))
    r = get_tweets_weights_by_ids(indexer, words_weights)
    print(r)
