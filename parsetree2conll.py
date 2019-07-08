from nltk.tree import Tree
from nltk.corpus import treebank as tb

penni_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
              'POS', 'PRP', 'PRP$', 'RB', 'SYM', 'TO', 'UH', 'VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'WDT', 'WP',
              'WP$', 'WRB', ',', '.', ':', '(', ')', '-NONE-']


def getFirstLvNPsOfParseTree(parse_tree, nps, display_tree=False):
    if display_tree:
        Tree.pretty_print(parse_tree)
        # print(Tree.leaf_treeposition(parser_tree, 1)) get a child index by leaves list index
        # print(parser_tree[(0, 0, 1,)]) get a tree by index
    for subtree in parse_tree:
        if isinstance(subtree, Tree) and subtree.label() == 'NP':
            np = subtree
            start_flag = "B-NP"
            print('\nNP: '+' '.join(Tree.leaves(np)))
            # may or may not be a terminal
            for np_derivation in Tree.subtrees(np):
                # below gets smaller np scope
                # getNPsOfParseTree(np_derivation, nps, False)
                if np_derivation.label() in penni_tags:
                    print(np_derivation.leaves()[0]+'\t'+np_derivation.label()+'\t'+start_flag)
                    start_flag = "I-NP"
            nps.append(Tree.leaves(np))
        elif isinstance(subtree, Tree) and subtree.label() != 'NP':
            getFirstLvNPsOfParseTree(subtree, nps, False)
        else:
            # reach terminal
            pass


def getSecondLvNPsOfParseTree(parse_tree, nps, display_tree=False):
    if display_tree:
        Tree.pretty_print(parse_tree)

    for subtree in parse_tree:
        if isinstance(subtree, Tree) and subtree.label() == 'NP' and subtree.height() == 3:
            np = subtree
            start_flag = "B-NP"
            print('\nNP: ' + ' '.join(Tree.leaves(np)))
            # obtained = False
            # may or may not be a terminal
            for np_derivation in Tree.subtrees(np):
                getSecondLvNPsOfParseTree(np_derivation, nps, False)
                if np_derivation.label() in penni_tags:
                    # if not obtained:
                    #     print('\nNP: ' + ' '.join(Tree.leaves(np)))
                    #     nps.append(Tree.leaves(np))
                    #     obtained = True
                    print(np_derivation.leaves()[0]+'\t'+np_derivation.label()+'\t'+start_flag)
                    start_flag = "I-NP"
            nps.append(Tree.leaves(np))
        elif isinstance(subtree, Tree) and subtree.label() != 'NP':
            getSecondLvNPsOfParseTree(subtree, nps, False)
        elif isinstance(subtree, Tree) and subtree.label() == 'NP' and subtree.height() != 3:
            getSecondLvNPsOfParseTree(subtree, nps, False)
        else:
            # reach terminal
            pass


def makeTrain(parser_tree, nps):
    words = parser_tree.leaves()
    # treepositions('leaves')  to get children indexes
    tags = [parser_tree[tag_i].label() for tag_i in [i[:-1] for i in parser_tree.treepositions('leaves')]]
    output_format = [[word, tag, 'O'] for word, tag in zip(words, tags)]

    cursor = 0
    print(nps)
    for np in nps:
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
                    break
            else:
                cursor += 1
                output_format[cursor][2] = tag
                tag = 'I-NP'
    print(output_format)
    return output_format


def outputCRFFormat(formatted, fileout):
    for line in formatted:
        fileout.write('\t'.join(line)+'\n')


# unnecessary to manage test in this way. just make it as same as training format
# just make it fit in gate diff tool format
def makeTestWithoutAnswers(parser_tree):
    words = parser_tree.leaves()
    tags = [parser_tree[tag_i].label() for tag_i in [i[:-1] for i in parser_tree.treepositions('leaves')]]
    return [[word, tag] for word, tag in zip(words, tags)]


def makeData(getNPsOfParseTree):
    train = open('../CRF++-0.58/train_lv2.data', 'w')
    test = open('../CRF++-0.58/test_lv2.conll', 'w')  # output answer.conll
    test_no_answers = open('../CRF++-0.58/test_no_golds_lv2.data', 'w')  # output answer_no_golds.conll
    for parsed_sent in tb.parsed_sents()[:2000]:
        res_nps = []
        getNPsOfParseTree(parsed_sent, res_nps, False)
        outputCRFFormat(makeTrain(parsed_sent, res_nps), train)

    for parsed_sent in tb.parsed_sents()[2000:]:
        res_nps = []
        getNPsOfParseTree(parsed_sent, res_nps, False)
        outputCRFFormat(makeTrain(parsed_sent, res_nps), test)

    for parsed_sent in tb.parsed_sents()[2000:]:
        outputCRFFormat(makeTestWithoutAnswers(parsed_sent), test_no_answers)

    train.close()
    test.close()
    test_no_answers.close()


if __name__ == '__main__':
    # 3914 sentences 100676 words, 2000 sentences(51959 tokens) for training
    # makeData(getFirstLvNPsOfParseTree)
    # makeData(getSecondLvNPsOfParseTree)
    #

    demo = open('./demo.data', 'w')
    for parsed_sent in tb.parsed_sents()[1:12]:
        res_nps = []
        getFirstLvNPsOfParseTree(parsed_sent, res_nps, True)
        print('param:  ', res_nps)
        outputCRFFormat(makeTrain(parsed_sent, res_nps), demo)
