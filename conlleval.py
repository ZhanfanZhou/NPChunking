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