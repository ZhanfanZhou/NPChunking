import pandas as pd


def sep(file_path):
    df = pd.read_csv(file_path, header=0, sep='\t', dtype={'tweet': str})
    print(df.head(10))
    out_IND = open('./out_IND.data', 'w', encoding='utf-8')
    out_GRP = open('./out_IND.data', 'w', encoding='utf-8')
    out_OTH = open('./out_IND.data', 'w', encoding='utf-8')
    for index, row in df.iterrows():
        # print("Tweet: ", row.tweet)
        if row.subtask_c == "IND":
            out_IND.write(row.tweet+'\n')
        elif row.subtask_c == "GRP":
            out_GRP.write(row.tweet + '\n')
        elif row.subtask_c == "OTH":
            out_OTH.write(row.tweet + '\n')
        print("finished.")


sep('./olid-training-v1.0.tsv')
