# coding: utf-8
base_folder = "probase/"
classes = ['business', 'tech']


# In[22]:


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


import numpy as np


# read probase
def get_prob_from_Probase():
    cls2word2prob = dict()
    max_entity_freq = 0

    for cls in classes:
        cls2word2prob[cls] = dict()
        with open("probase/probase_{}.txt".format(cls), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                words = line.split('\t')
                concpet = words[0]
                entity = words[1]
                freq = words[2]
                entity_freq = words[3]
                prob = float(freq) / float(entity_freq)
                cls2word2prob[cls][entity] = prob

                max_entity_freq = max(int(entity_freq), max_entity_freq)

    return cls2word2prob, max_entity_freq


import sys
import math


#         decision by probase
def byProbase(cls2word2prob, max_entity_freq, sentence):
    max_classes = ""
    max_sum = 0.0 - sys.float_info.max
    words = sentence.split(' ')

    smoothing = 1.0 / float(max_entity_freq)

    for cls in classes:
        log_sum = 0
        for word in words:
            if word == "":
                continue
            if word in cls2word2prob[cls]:
                log_sum += math.log(cls2word2prob[cls][word])
            # print("\t\t{}, {}:{}".format(cls, word, cls2word2prob[cls][word]))
            else:
                log_sum += math.log(smoothing)
        if log_sum > max_sum:
            max_sum = log_sum
            max_classes = cls
            #         print ("\t{}: {}".format(cls, log_sum))
    return max_classes


def process_cross_validation(num_fold):
    cls2word2prob, max_entity_freq = get_prob_from_Probase()

    threshold = 350
    #     threshold = 10
    gap = threshold / num_fold
    list_accuracy = []
    for i in range(num_fold):
        docs_train = []
        docs_test = []
        for j in range(len(classes)):
            docs_all = []
            for doc in read_documents(j, threshold):
                docs_all.append(doc)
            docs_test += docs_all[int(i * gap):int((i + 1) * gap)]
            if i == 0:
                docs_train += docs_all[int((i + 1) * gap):]
            else:
                docs_train += docs_all[0:int(i * gap)] + docs_all[int((i + 1) * gap):]
                #         print("Train/Test: {}/{}".format(len(docs_train), len(docs_test)))

        list_ours = []
        list_gold = []
        for doc in docs_test:
            #             print("\n" + doc.label)
            max_classes = byProbase(cls2word2prob, max_entity_freq, doc.sentence)
            list_gold.append(doc.label)
            list_ours.append(max_classes)

        acc = compute_accuracy(list_ours, list_gold)
        print("\t{}".format(acc))
        list_accuracy.append(acc)
    print(np.mean(list_accuracy))


process_cross_validation(5)

