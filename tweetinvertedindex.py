import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer
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
        self.words_weights = []
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    def lookup(self, word, stem=True):
        """
        Lookup a word in the index
        :param stem prevent stem(stemmed word) != stemmed word
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
        '''
        in order to get words that are mostly OFF/NOT,
        give each word a score by going through inverted index
        then sort it
        :return:
        '''
        iidx = self.index
        label_info = self.info

        def getPorprotion(ids):
            l = len(ids)
            off = 0
            for id in ids:
                if label_info[id] == 'OFF':
                    off += 1
            s = -math.log(l, 2) * (pow((off / float(l) - 0.5), 2))
            return s

        res = sorted([(word, getPorprotion(docs)) for word, docs in iidx.items()],
                     key=lambda x: x[1])
        return res

    def init_words_weights(self):
        '''
        get "weights" for each word by going through inverted index
        :return: (word, weight)
        '''
        iidx = self.index
        label_info = self.info

        def getPorprotion(ids):
            l = len(ids)
            off = 0
            for id in ids:
                if label_info[id] == 'OFF':
                    off += 1
            return off / float(l)
        self.words_weights = [(word, getPorprotion(docs)) for word, docs in iidx.items()]

    def getExceptionals(self, p_words, n=200):
        '''
        get not/offensive tweets that contains words that are mostly offensive/not.
        :param p_words: polarized words from getPolarizedWord()
        :param n: top n words
        :return: none
        '''
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

    def get_tweets_weights_by_ids(self, ids):
        """
        :return: a list of scores that ordered by param ids
        """
        if not self.words_weights:
            self.init_words_weights()
        ranks = {}
        for word, weight in self.words_weights:
            for id in self.index.get(word):
                ranks[id] = ranks.get(id, 0) + weight

        scores = []
        for id in ids:
            scores.append([ranks.get(id)])
        return scores

    def get_test_weights(self, test_tweets, test_ids):
        """
        first transform words_weights into a dictionary
        :param test_tweets:
        :param test_ids:
        :return: see get_tweets_weights_by_ids()
        """
        if not self.words_weights:
            self.init_words_weights()
        weights_dict = {}
        for word, weight in self.words_weights:
            weights_dict[word] = weight
        ranks = {}
        for id, tweet in zip(test_ids, test_tweets):
            for token in self.tokenizer(tweet):
                token = token.lower()
                if self.stemmer:
                    token = self.stemmer.stem(token)
                weight = weights_dict.get(token, None)
                if weight is not None:
                    ranks[id] = ranks.get(id, 0) + weight

        scores = []
        for id in test_ids:
            scores.append([ranks.get(id)])
        return scores


def createTweetsInvertedIndex(inverted_indexer, tweet_path='./olid-training-v1.0.tsv'):
    '''
    create a inverted index for tweets data.
    :param inverted_indexer:
    :param tweet_path:
    :return:
    '''
    df = pd.read_csv(tweet_path, header=0, sep='\t', dtype={'id': str})
    for index, row in df.iterrows():
        inverted_indexer.add(row.tweet, row.id, row.subtask_a)
    return inverted_indexer


if __name__ == '__main__':
    indexer = createTweetsInvertedIndex(Index(tokenize,
                                              EnglishStemmer(),
                                              nltk.corpus.stopwords.words('english')))
    ranked_words_weights = indexer.getPolarizedWord()
    indexer.getExceptionals(ranked_words_weights)
    # print(indexer.lookup('fuck', stem=True))
