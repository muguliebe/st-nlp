#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import os

from PIL import Image
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

## For Morph Analyze
from konlpy.tag import Mecab

## For Name-Entity Recognition
from py_aho_corasick import py_aho_corasick

## For TF-IDF
import math

documents = {}
doc_idx = 0


def add_doc(dict):
    global doc_idx
    documents[doc_idx] = dict
    doc_idx = doc_idx + 1


def idf(term):
    num = 0
    for i in range(doc_idx):
        # print i
        dict = documents[i]
        if term in dict:
            # print 'ok' , i
            num = num + 1
    if num == 0:
        return 0
    tmp = (doc_idx * 1.0 / num)
    # print tmp, doc_idx, num

    if (tmp == 0):
        return 0
    else:
        return math.log(tmp)


d = path.dirname(__file__)
mask = np.array(Image.open(path.join(d, "alice_mask.png")))


def starter():
    text = "The fifth-generation network is a key factor for up-and-coming data-oriented technologies." \
           "Virtual reality, self-driving automobiles, artificial intelligence and the Internet of Things, " \
           "which will be at the forefront of future industries, require a network that can transmit massive data seamlessly." \
           "To keep pace with the future connected world, Korean telecom operators KT, SK Telecom and LG Uplus are speeding up the development " \
           "of the fifth-generation mobile network, or 5G network, with the aim of commercializing it by 2019.  " \
           "According to the International Telecommunication Union, a 5G network must promise download speeds of up to 20 Gbps (billions of bits per second). The network should support" \
           " at least 1 million connected devices per square kilometer with a response time of less than 0.001 second and also " \
           "support mobile devices moving at up to 500 kilometers per hour. Over the past year, the three telecom operators have demonstrated the " \
           "5G network as pilot projects using both the 3.5 gigahertz low frequency band and the 28 GHz high frequency band. "

    dict = {}
    dict[u'SK텔레콤'] = 10
    dict[u'통신'] = 5
    dict[u'AI'] = 3

    wordcloud = WordCloud(background_color="white", font_path=path.join(d + 'fonts', "NanumMyeongjo.ttf"))
    # wordcloud = WordCloud(background_color="white", mask=mask, font_path=path.join(d + 'fonts', "NanumPen.ttf"))
    # wordcloud.fit_words(f)
    wordcloud.generate(text)

    wordcloud.to_file(path.join(d, "wc.png"))
    utils.send_image(path.join(d, "wc.png"))


def wc_by_word(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f.readlines():
            sp = line.decode('utf-8').strip().split(" ")
            for word in sp:
                if len(word) < 1: continue
                if word not in dict:
                    dict[word] = 1
                else:
                    dict[word] = dict[word] + 1

    wordcloud = WordCloud(background_color="white", font_path=path.join(d + 'fonts', "NanumMyeongjo.ttf"))
    wordcloud.fit_words(dict)

    wordcloud.to_file(path.join(d, "wc.png"))
    utils.send_image(path.join(d, "wc.png"))


mecab = Mecab()
tag_list = ["NNG", "NNP", "SL"]


def wc_by_morph(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f.readlines():
            result = mecab.pos(line.decode("utf-8").strip())
            for r in result:
                word = r[0]
                tag = r[1]
                if tag not in tag_list: continue
                if len(word) < 1: continue

                if word not in dict:
                    dict[word] = 1
                else:
                    dict[word] = dict[word] + 1

    wordcloud = WordCloud(background_color="white", font_path=path.join(d + 'fonts', "NanumMyeongjo.ttf"))
    wordcloud.fit_words(dict)

    wordcloud.to_file(path.join(d, "wc.png"))
    utils.send_image(path.join(d, "wc.png"))


def wc_by_ner(file_name):
    entities = [(u'SK텔레콤', u'COMPANY'), (u'5G', u'TERM')]
    with open("wiki_entity_100000.txt", "r") as f:
        for line in f.readlines():
            line = line.decode('utf-8').strip()
            sp = line.split("\t")
            word = sp[0]
            tag = sp[1]
            # print word
            t = (word, tag)
            entities.append(t)

    B = py_aho_corasick.Automaton(entities)
    dict = {}
    with open(file_name) as f:
        for line in f.readlines():
            line_entity = []
            line_word = []
            line_concat = line.decode("utf-8").replace(" ", "")
            for idx, k, v in B.get_keywords_found(line_concat):
                # print idx, k, v
                line_entity.append(k.upper())

            result = mecab.pos(line.decode("utf-8").strip())
            for r in result:
                flag = False
                word = r[0].upper()
                tag = r[1]
                if tag not in tag_list: continue
                if len(word) < 1: continue
                for entity in line_entity:
                    if word in entity:
                        flag = True
                        continue

                if (flag == False):
                    line_word.append(word)
                    # print word

            for word in line_entity:
                word = word.upper()
                if word not in dict:
                    dict[word] = 1
                else:
                    dict[word] = dict[word] + 1

            for word in line_word:
                word = word.upper()
                if word not in dict:
                    dict[word] = 1
                else:
                    dict[word] = dict[word] + 1

    wordcloud = WordCloud(background_color="white", font_path=path.join(d + 'fonts', "NanumMyeongjo.ttf"))
    wordcloud.fit_words(dict)

    wordcloud.to_file(path.join(d, "wc.png"))
    utils.send_image(path.join(d, "wc.png"))


def wc_by_morph_folder(folder_name):
    files = os.listdir(folder_name)
    dict = {}
    # file_name = ''
    for file in files:
        file_name = folder_name + "/" + file
        with open(file_name) as f:
            for line in f.readlines():
                result = mecab.pos(line.decode("utf-8").strip())
                for r in result:
                    word = r[0]
                    tag = r[1]
                    if tag not in tag_list: continue
                    if len(word) < 1: continue

                    if word not in dict:
                        dict[word] = 1
                    else:
                        dict[word] = dict[word] + 1

    wordcloud = WordCloud(background_color="white", font_path=path.join(d + 'fonts', "NanumMyeongjo.ttf"))
    wordcloud.fit_words(dict)

    wordcloud.to_file(path.join(d, "wc.png"))
    utils.send_image(path.join(d, "wc.png"))


def wc_by_word_folder(folder_name):
    files = os.listdir(folder_name)
    dict = {}
    # file_name = ''
    for file in files:
        file_name = folder_name + "/" + file
        with open(file_name) as f:
            for line in f.readlines():
                sp = line.decode('utf-8').strip().split(" ")
                for word in sp:
                    if len(word) < 1: continue
                    if word not in dict:
                        dict[word] = 1
                    else:
                        dict[word] = dict[word] + 1

    wordcloud = WordCloud(background_color="white", font_path=path.join(d + 'fonts', "NanumMyeongjo.ttf"))
    wordcloud.fit_words(dict)

    wordcloud.to_file(path.join(d, "wc.png"))
    utils.send_image(path.join(d, "wc.png"))


def wc_by_ner_folder(folder_name):
    entities = [(u'SK텔레콤', u'COMPANY'), (u'5G', u'TERM')]

    with open("wiki_entity_100000.txt", "r") as f:
        for line in f.readlines():
            line = line.decode('utf-8').strip()
            sp = line.split("\t")
            word = sp[0]
            tag = sp[1]
            # print word
            t = (word, tag)
            entities.append(t)

    B = py_aho_corasick.Automaton(entities)

    files = os.listdir(folder_name)
    dict = {}
    # file_name = ''
    for file in files:
        file_name = folder_name + "/" + file

        with open(file_name) as f:
            for line in f.readlines():
                line_entity = []
                line_word = []
                line_concat = line.decode("utf-8").replace(" ", "")
                for idx, k, v in B.get_keywords_found(line_concat):
                    # print idx, k, v
                    line_entity.append(k.upper())

                result = mecab.pos(line.decode("utf-8").strip())
                for r in result:
                    flag = False
                    word = r[0].upper()
                    tag = r[1]
                    if tag not in tag_list: continue
                    if len(word) < 1: continue
                    for entity in line_entity:
                        if word in entity:
                            flag = True
                            continue

                    if (flag == False):
                        line_word.append(word)
                        # print word

                for word in line_entity:
                    word = word.upper()
                    if word not in dict:
                        dict[word] = 1
                    else:
                        dict[word] = dict[word] + 1

                for word in line_word:
                    word = word.upper()
                    if word not in dict:
                        dict[word] = 1
                    else:
                        t
                        dict[word] = dict[word] + 1

    wordcloud = WordCloud(background_color="white", font_path=path.join(d + 'fonts', "NanumMyeongjo.ttf"))
    wordcloud.fit_words(dict)

    wordcloud.to_file(path.join(d, "wc.png"))
    utils.send_image(path.join(d, "wc.png"))


def wc_by_tfidf(doc_name, folder_name):
    pass


if __name__ == '__main__':
    starter()
    # wc_by_word("data/17670_N3.txt")
    # wc_by_word_folder("data")
    # wc_by_morph("data/17670_N4.txt")
    # wc_by_morph_folder("data")
    # wc_by_ner("data/17670_N1.txt")
    # wc_by_ner_folder("data")
    # wc_by_tfidf("data/17670_N1.txt","data")


