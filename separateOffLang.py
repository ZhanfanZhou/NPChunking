# this file is for bettering looking into the data by separating the tweets by classes.
import pandas as pd


def sep(file_path, raw_tweet=False, gate=False):
    """
    :param file_path:
    :param raw_tweet: false for tweetlda.py use, true for obtaining raw tweets
    :param gate output file for gate "tweet_splitting"(from Parsa :D)
    :return:
    """
    df = pd.read_csv(file_path, header=0, sep='\t', dtype={'id': str})
    # print(df.head(10))
    if not raw_tweet:
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
    elif not gate:
        out_NOT = open('./out_NOT_text.data', 'w', encoding='utf-8')
        out_UNT = open('./out_UNT_text.data', 'w', encoding='utf-8')
        out_TIN = open('./out_TIN_text.data', 'w', encoding='utf-8')
        out_IND = open('./out_IND_text.data', 'w', encoding='utf-8')
        out_GRP = open('./out_GRP_text.data', 'w', encoding='utf-8')
        out_OTH = open('./out_OTH_text.data', 'w', encoding='utf-8')
        for index, row in df.iterrows():
            if row.subtask_a == "NOT":
                out_NOT.write(row.tweet + '\n')
            else:
                if row.subtask_b == "UNT":
                    out_UNT.write(row.tweet + '\n')
                else:
                    out_TIN.write(row.tweet + '\n')
                    if row.subtask_c == "IND":
                        out_IND.write(row.tweet+'\n')
                    elif row.subtask_c == "GRP":
                        out_GRP.write(row.tweet + '\n')
                    elif row.subtask_c == "OTH":
                        out_OTH.write(row.tweet + '\n')
    else:
        out_NOT = open('./out_NOT_gate.data', 'w', encoding='utf-8')
        out_UNT = open('./out_UNT_gate.data', 'w', encoding='utf-8')
        out_TIN = open('./out_TIN_gate.data', 'w', encoding='utf-8')
        out_IND = open('./out_IND_gate.data', 'w', encoding='utf-8')
        out_GRP = open('./out_GRP_gate.data', 'w', encoding='utf-8')
        out_OTH = open('./out_OTH_gate.data', 'w', encoding='utf-8')
        out_all = open('./out_all_a_gate.data', 'w', encoding='utf-8')
        for index, row in df.iterrows():
            out_all.write(row.id + '\t' + row.subtask_a + '\t' + row.tweet + '\n')
            if row.subtask_a == "NOT":
                out_NOT.write(row.id + '\t' + row.subtask_a + '\t' + row.tweet + '\n')
            else:
                if row.subtask_b == "UNT":
                    out_UNT.write(row.id + '\t' + row.subtask_b + '\t' + row.tweet + '\n')
                else:
                    out_TIN.write(row.id + '\t' + row.subtask_b + '\t' + row.tweet + '\n')
                    if row.subtask_c == "IND":
                        out_IND.write(row.id + '\t' + row.subtask_c + '\t' + row.tweet + '\n')
                    elif row.subtask_c == "GRP":
                        out_GRP.write(row.id + '\t' + row.subtask_c + '\t' + row.tweet + '\n')
                    elif row.subtask_c == "OTH":
                        out_OTH.write(row.id + '\t' + row.subtask_c + '\t' + row.tweet + '\n')

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


def make_test(file_lb='./labels-levela.csv', file_twt='./testset-levela.tsv'):
    df_lb = pd.read_csv(file_lb, header=0, sep=',', dtype={'id': str}).set_index('id')
    df_twt = pd.read_csv(file_twt, header=0, sep='\t', dtype={'id': str}).set_index('id')
    # print(df_lb.head(10))
    # print(df_twt.head(10))
    out_test = open('./test.data', 'w', encoding='utf-8')
    df = df_twt.join(df_lb, on='id').reset_index()
    df = df.rename(columns={'label': 'subtask_a'})
    return df


if __name__ == '__main__':
    # sep('./olid-training-v1.0.tsv', raw_tweet=True, gate=True)
    make_test()
