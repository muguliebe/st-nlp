# -*- coding: utf-8 -*-
from py_aho_corasick import py_aho_corasick


def load_entity(filename):
    content = []
    with open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            tt = line.split('\t')
            word = tt[0].replace("\n", "")
            tag = tt[1].replace("\n", "")
            content.append((word, tag))
    return content


# keywords only
# entity_dict = [(u'SK텔레콤',u'COMPANY'), (u'5G',u'TERM')]

entity_dict = load_entity("wiki_entity_100000.txt")
A = py_aho_corasick.Automaton(entity_dict)

with open("data/17670_N1.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        line_concat = line.decode("utf-8").replace(" ", "")
        # print line
        for idx, k, v in A.get_keywords_found(line_concat):
            print
            idx, k.encode('utf-8'), v



