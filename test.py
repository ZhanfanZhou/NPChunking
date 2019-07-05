# This script converts a .tsv or .csv file(separated by tab) into CoNLL format.
# The input .tsv file must include five columns:
# tweets content, begin offset, end offset, target extraction, and extraction type.

# Note that the converter is based on char offset which is necessary to provide in input file.
# Input tweets are duplicated,and each duplicate corresponds only with an extraction of its own.

# May,2019

import nltk
from init import outputCRFFormat
import pandas as pd


def makeTrain(sent, nps):
    words = nltk.word_tokenize(sent)
    # print('tokenize whole sent: ', words)
    tags = [t[1] for t in nltk.pos_tag(words)]
    output_format = [[word, tag, 'O'] for word, tag in zip(words, tags)]

    cursor = 0
    print('working on: ', nps)
    for np in nps:
        print('====tagging====', np)
        tag = 'B-NP'
        for word_index in range(len(np)):
            if word_index == 0:
                possible_index = list(filter(lambda x: x >= cursor,
                                             [i for i, x in enumerate(words) if x == np[word_index]]))
                # possible index may be empty
                if possible_index:
                    index = possible_index[0]
                    output_format[index][2] = tag
                    tag = 'I-NP'
                    cursor = index
                else:
                    print('labeling failed!!')
                    break
            else:
                cursor += 1
                output_format[cursor][2] = tag
                tag = 'I-NP'
    print(output_format)
    return output_format


def readin(files):
    def f(x):
        return pd.Series(dict(begin=list(x['begin']),
                              end=list(x['end']),
                              type=list(x['type']),
                              extraction=list(x['extraction']),
                              tweet=x['tweet']))
    for f_path in files:
        # see DataFrame.apply
        # https://stackoverflow.com/questions/17841149/pandas-groupby-how-to-get-a-union-of-strings
        df = pd.read_csv(f_path, header=0, sep='\t', dtype={'tweet': str})\
            .groupby('tweet_id')\
            .apply(f)\
            .reset_index()
        # print(df.head(10))
        for index, row in df.iterrows():
            print("Tweet: ", row.tweet.values[0])
            adrs = list(collectADRs(row))
            outputCRFFormat(makeTrain(row.tweet.values[0], adrs), demo)
            print()


def collectADRs(row):
    # rank them first, then output spans
    span_tokens = []
    for span in row.extraction:
        if pd.isna(span):
            span_tokens.append([])
        else:
            span_tokens.append(nltk.word_tokenize(span))
    offsets = sorted(zip(row.begin, row.end, span_tokens), key=lambda x: x[0])
    print(offsets)
    for offset in offsets:
        yield offset[2]


demo = open('./demo.data', 'w')
# readin(['~/zhanfan/SMM4H_data/TrainData1.tsv',
#         '~/zhanfan/SMM4H_data/TrainData2.tsv',
#         '~/zhanfan/SMM4H_data/TrainData3.tsv',
#         '~/zhanfan/SMM4H_data/TrainData4.tsv'])

readin(['~/zhanfan/SMM4H_data/TrainData4.tsv'])
