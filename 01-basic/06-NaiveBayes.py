# coding: utf-8

# In[69]:


base_folder = "data2/"
classes = ['business', 'entertainment', 'politics', 'sport', 'tech']


def read_wordlist():
    word_list = []
    with open("data2/word_list.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            word_list.append(line)
    return word_list


class Doc():
    def __init__(self, label, sentence):
        self.label = label
        self.sentence = sentence


def read_documents(idx_class, threshold_doc_num):
    fname = base_folder + classes[idx_class] + '.txt'
    docs = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            d = Doc(classes[idx_class], line)
            docs.append(d)
            if len(docs) >= threshold_doc_num:
                break
    return docs


def compute_accuracy(list_ours, list_gold):
    tp = 0.0
    total = float(len(list_gold))
    for i in range(len(list_ours)):
        if list_ours[i] == list_gold[i]:
            tp += 1.0
    acc = tp / total
    return acc


def get_cls2word2prob(docs):
    cls2word2prob = dict()
    cls2cnt = dict()
    word_list = read_wordlist()

    #############################################
    ToDo = "Generate cls2word2prob dictionary"
    # key: class --> value: dict(key: word --> value: probability)
    # implement smoothing
    #############################################

    return cls2word2prob


import math
import sys


def NaiveBayes(cls2word2prob, sentence):
    max_classes = ""
    max_sum = 0.0 - sys.float_info.max
    words = sentence.split(' ')

    #############################################
    ToDo = "Bayes classifier"
    #############################################

    return max_classes


def process_simple():
    docs_train = []
    docs_test = []

    threshold_train = 350
    threshold_test = 10
    for i in range(len(classes)):
        for doc in read_documents(i, threshold_train):
            docs_train.append(doc)
        for doc in read_documents(i, threshold_test):
            docs_test.append(doc)

    cls2word2prob = get_cls2word2prob(docs_train)

    list_ours = []
    list_gold = []
    for doc in docs_test:
        max_classes = NaiveBayes(cls2word2prob, doc.sentence)
        list_gold.append(doc.label)
        list_ours.append(max_classes)
    # print("Gold: {} --> Max: {}".format(t.label, max_classes))

    print("Accuracy: {}".format(compute_accuracy(list_ours, list_gold)))


process_simple()

# In[72]:


import numpy as np


def process_cross_validation(num_fold):
    threshold = 350
    gap = threshold / num_fold
    list_accuracy = []
    for i in range(num_fold):
        docs_train = []
        docs_test = []

        #############################################
        ToDo = "Seperate train/test dataset"
        #############################################

        cls2word2prob = get_cls2word2prob(docs_train)
        list_ours = []
        list_gold = []
        for doc in docs_test:
            max_classes = NaiveBayes(cls2word2prob, doc.sentence)
            list_gold.append(doc.label)
            list_ours.append(max_classes)

        acc = compute_accuracy(list_ours, list_gold)
        print("\t{}".format(acc))
        list_accuracy.append(acc)
    print(np.mean(list_accuracy))


process_cross_validation(5)