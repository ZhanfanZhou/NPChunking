import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# import seaborn as sb
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from twokenize import tokenize
from textblob import TextBlob
import nltk

# file_path = "./olid-training-v1.0.tsv"
file_path = "out_NOT.data"
raw_data = pd.read_csv(file_path, header=0, sep='\t', dtype={'tweet': str})
reindexed_data = raw_data['tweet']
# reindexed_data.index = raw_data['id']
print(reindexed_data.head())


def get_top_n_words1(n_top_words, count_vectorizer, text_data):
    '''
    returns a tuple of the top n words in a sample and their
    accompanying counts, given a CountVectorizer object and text sample
    '''
    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0, :], 1)
    word_values = np.flip(np.sort(vectorized_total)[0, :], 1)

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i, word_indices[0, i]] = 1

    words = [word[0].encode('ascii').decode('utf-8') for
             word in count_vectorizer.inverse_transform(word_vectors)]

    return words, word_values[0, :n_top_words].tolist()[0]


new_stop_words = nltk.corpus.stopwords.words('english')+[",",".","!","?","&",";","\"","'",':']
count_vectorizer = CountVectorizer(stop_words=new_stop_words, tokenizer=tokenize)
words, word_values = get_top_n_words1(n_top_words=20,
                                      count_vectorizer=count_vectorizer,
                                      text_data=reindexed_data)

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(range(len(words)), word_values)
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words, rotation='vertical')
ax.set_title('Top words in tweet (excluding stop words)')
ax.set_xlabel('Word')
ax.set_ylabel('Number of occurences')
plt.show()

word_feature = 200
small_count_vectorizer = CountVectorizer(stop_words='english', max_features=word_feature)
small_text_sample = reindexed_data.sample(n=300, random_state=0).values
small_document_term_matrix = small_count_vectorizer.fit_transform(small_text_sample)
# print("doc", small_document_term_matrix)
# print("doc", small_document_term_matrix.shape)
# scipy.sparse.csr.csr_matrix


def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    print("KEYS", keys)
    return keys


def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    print(count_pairs)
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


def get_top_n_words2(n, keys, document_term_matrix, count_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        print("current topic ", topic)
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                # print("what is i?", i)
                # print("adding", document_term_matrix[i, :])
                temp_vector_sum += document_term_matrix[i]
        # print("temp_vector_sum added", temp_vector_sum)
        if isinstance(temp_vector_sum, int):
            print("all zeros!!")
            temp_vector_sum = np.zeros((1, word_feature))
        else:
            temp_vector_sum = temp_vector_sum.toarray()
        print("-------")
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:], 0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words


n_topics = 8
lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(small_document_term_matrix)
lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
top_n_words_lsa = get_top_n_words2(10, lsa_keys, small_document_term_matrix, small_count_vectorizer)

for i in range(len(top_n_words_lsa)):
    print("Topic {}: ".format(i+1), top_n_words_lsa[i])

top_3_words = get_top_n_words2(3, lsa_keys, small_document_term_matrix, small_count_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(lsa_categories, lsa_counts)
ax.set_xticks(lsa_categories)
ax.set_xticklabels(labels)
ax.set_ylabel('Number of tweets')
ax.set_title('LSA topic counts')
plt.show()

print("\n======LDA======\n")

lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online',
                                      random_state=0, verbose=0)
lda_topic_matrix = lda_model.fit_transform(small_document_term_matrix)
lda_keys = get_keys(lda_topic_matrix)
lda_categories, lda_counts = keys_to_counts(lda_keys)
top_n_words_lda = get_top_n_words2(10, lda_keys, small_document_term_matrix, small_count_vectorizer)

for i in range(len(top_n_words_lda)):
    print("Topic {}: ".format(i+1), top_n_words_lda[i])

top_4_words = get_top_n_words2(4, lda_keys, small_document_term_matrix, small_count_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_4_words[i] for i in lda_categories]

fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(lda_categories, lda_counts)
ax.set_xticks(lda_categories)
ax.set_xticklabels(labels)
ax.set_title('LDA topic counts')
ax.set_ylabel('Number of tweets')
plt.show()
