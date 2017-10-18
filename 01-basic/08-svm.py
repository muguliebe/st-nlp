# coding: utf-8
base_folder = "data2/"
classes = ['business', 'entertainment', 'politics', 'sport', 'tech']


# In[10]:


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


# In[77]:


from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

threshold_doc_num = 350


# 각 클래스별 상위 top-n word
def get_words_for_features(top_n):
    stop = stopwords.words('english')
    word2cnt = defaultdict(int)
    list_sentences_all = []
    for i in range(len(classes)):
        docs = read_documents(i, threshold_doc_num)
        sentences = ''
        for doc in docs:
            sentences += doc.sentence + ' '
        list_sentences_all.append(sentences)

    tfidf = TfidfVectorizer(stop_words='english', min_df=1)
    response = tfidf.fit_transform(list_sentences_all)
    feature_array = np.array(tfidf.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    top_features = feature_array[tfidf_sorting][:top_n]

    return top_features


def train_SVM(topwords, docs_train):
    set_topwords = set(topwords)

    X = []
    Y = []
    for doc in docs_train:
        label_idx = classes.index(doc.label)
        words = doc.sentence.split(' ')
        word2cnt = defaultdict(float)
        num_total = float(len(words))
        for word in words:
            if word in set_topwords:
                word2cnt[word] += 1.0

        x = []
        for word in topwords:
            x.append(word2cnt[word] / num_total)

        X.append(x)
        Y.append(label_idx)

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)
    return lin_clf


def compute_accuracy(list_ours, list_gold):
    tp = 0.0
    total = float(len(list_gold))
    for i in range(len(list_ours)):
        if list_ours[i] == list_gold[i]:
            tp += 1.0
    acc = tp / total
    return acc


def process_cross_validation(num_fold):
    threshold = 350
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

        num_dim = 300
        topwords = get_words_for_features(num_dim)

        set_topwords = set(topwords)
        lin_clf = train_SVM(topwords, docs_train)
        list_ours = []
        list_gold = []
        for doc in docs_test:
            words = doc.sentence.split(' ')
            word2cnt = defaultdict(float)
            num_total = float(len(words))
            for word in words:
                if word in set_topwords:
                    word2cnt[word] += 1.0
            x = []
            for word in topwords:
                x.append(word2cnt[word] / num_total)

            Y = lin_clf.predict([x])
            max_classes = classes[Y[0]]

            list_gold.append(doc.label)
            list_ours.append(max_classes)

        acc = compute_accuracy(list_ours, list_gold)
        print("\t{}".format(acc))
        list_accuracy.append(acc)
    print(np.mean(list_accuracy))


process_cross_validation(5)

