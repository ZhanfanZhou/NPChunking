# this file is to
import pandas as pd


def sep(file_path):
    df = pd.read_csv(file_path, header=0, sep='\t', dtype={'id': str})
    # print(df.head(10))
    out_NOT = open('./out_NOT.data', 'w', encoding='utf-8')
    out_UNT = open('./out_UNT.data', 'w', encoding='utf-8')
    out_TIN = open('./out_TIN.data', 'w', encoding='utf-8')
    out_IND = open('./out_IND.data', 'w', encoding='utf-8')
    out_GRP = open('./out_GRP.data', 'w', encoding='utf-8')
    out_OTH = open('./out_OTH.data', 'w', encoding='utf-8')
    add_headers([out_NOT, out_UNT, out_TIN, out_IND, out_GRP, out_OTH])
    for index, row in df.iterrows():
        # print("Tweet: ", row.tweet)
        if row.subtask_a == "NOT":
            out_NOT.write(row.id+'\t'+row.tweet + '\n')
        else:
            if row.subtask_b == "UNT":
                out_UNT.write(row.id+'\t'+row.tweet + '\n')
            else:
                out_TIN.write(row.id+'\t'+row.tweet + '\n')
                if row.subtask_c == "IND":
                    out_IND.write(row.id+'\t'+row.tweet+'\n')
                elif row.subtask_c == "GRP":
                    out_GRP.write(row.id+'\t'+row.tweet + '\n')
                elif row.subtask_c == "OTH":
                    out_OTH.write(row.id+'\t'+row.tweet + '\n')

    print("finished.")
    out_NOT.close()
    out_UNT.close()
    out_TIN.close()
    out_IND.close()
    out_GRP.close()
    out_OTH.close()


def add_headers(outputs):
    for output in outputs:
        output.write('id\ttweet\n')


sep('./olid-training-v1.0.tsv')
