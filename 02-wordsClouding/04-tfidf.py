#-*- coding: utf-8 -*-

import math
import os

## For Morph Analyze
from konlpy.tag import Mecab

documents = {}
doc_idx = 0
def add_doc(dict):
    global doc_idx
    documents[doc_idx] = dict
    doc_idx = doc_idx + 1

def idf(term):
    num = 0
    for i in range(doc_idx):
        dict = documents[i]
        if term in dict:
            print 'ok' , i
            num = num + 1
    if num == 0:
        return 0
    tmp = (doc_idx * 1.0 / num )
    print tmp, doc_idx, num

    if (tmp == 0):
        return 0
    else:
        return math.log(tmp)

files = os.listdir("data")
mecab = Mecab()
tag_list = ["NNG", "NNP", "SL"]

# file_name = ''
for file in files:
    file_name = "data" + "/" + file
    dict = {}
    with open(file_name) as f:
        for line in f.readlines():
            dict = line.strip().split(" ")
            add_doc(dict)
            pass
        print(documents)
        # add_doc(dict)

print idf(u"sk")