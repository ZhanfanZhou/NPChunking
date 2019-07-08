# this script is to evaluate a conll format file.
# "answer_no_golds_lv2.conll..." is a file from gate diff tool which has already done the evaluation
# "answer_small..." is a subset from the previous one in case that gate has visualization problem for a large file.
# so it there are 2 things remained:
# post analyse the diff tool file, and write your own evaluation instead of using gate diff tool
# Obviously neither of them are done.
arr = [
{'a':1,'f':True},
{'a':2,'f':False},
{'a':3,'f':True},
{'a':4,'f':False},
{'a':5,'f':True}
]
print(min(filter(lambda x: x[1]['f'], tuple(enumerate(arr))), key=lambda x: x[1]['a'])[0])
import nltk

t = nltk.word_tokenize("The cat is lovely. Montreal is a city of Canada.")
print(nltk.pos_tag(t))