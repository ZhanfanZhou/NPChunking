import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer  # Assuming we're working with English
import pandas as pd
import math


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

    def lookup(self, word):
        """
        Lookup a word in the index
        """
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)

        return [(self.documents.get(id, None), id) for id in self.index.get(word)]

    def add(self, document, doc_id, label):
        """
        Add a document string to the index
        """
        self.__unique_id = doc_id
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if token in self.stopwords:
                continue

            if self.stemmer:
                token = self.stemmer.stem(token)

            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)

        self.documents[self.__unique_id] = document
        self.info[self.__unique_id] = label

#       show tweet unique id and contents as well


def createTweetsInvertedIndex(inverted_indexer, tweet_path='./olid-training-v1.0.tsv'):
    df = pd.read_csv(tweet_path, header=0, sep='\t', dtype={'id': str})
    for index, row in df.iterrows():
        inverted_indexer.add(row.tweet, row.id, row.subtask_a)


# two params are attributes of class index
def getPolarizedWord(iidx, label_info):
    def getPorprotion(ids):
        off = 0
        for id in ids:
            if label_info[id] == 'OFF':
                off += 1
        return off/float(len(ids)), len(ids)

    return sorted([(word, getPorprotion(docs)) for word, docs in iidx.items()], key=lambda x: -math.log(x[1][1], 2)*(pow((x[1][0]-0.5), 2)))


indexer = Index(nltk.word_tokenize,
              EnglishStemmer(),
              nltk.corpus.stopwords.words('english'))
createTweetsInvertedIndex(indexer)
print(getPolarizedWord(indexer.index, indexer.info))

