# coding: utf-8

# In[2]:


base_folder = "./data2/"
classes = ['business', 'entertainment', 'politics', 'sport', 'tech']

# In[41]:


from nltk.metrics import edit_distance
from nltk.collocations import BigramCollocationFinder
import nltk


def get_tokens_from_all_corpus():
    all_tokens = []
    for i in range(len(classes)):
        fname = base_folder + classes[i] + '.txt'
        with open(fname, encoding='utf-8') as f:
            for line in f:
                all_tokens += nltk.word_tokenize(line.strip())
    return all_tokens


def get_word2next_prob_sorted():
    all_tokens = get_tokens_from_all_corpus()
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens)

    scored = finder.score_ngrams(bigram_measures.raw_freq)
    word2next_prob = dict()
    for w1w2, prob in scored:
        w1, w2 = w1w2
        if w1 not in word2next_prob:
            word2next_prob[w1] = []
        word2next_prob[w1].append((w2, prob))
    for word in word2next_prob:
        sorted(word2next_prob[word], key=lambda tup: (-tup[1], tup[0]))
    return word2next_prob


def find_candidates_given_word(word2next_prob, last_word, target_word):
    candidates_probs = []
    for word, prob in word2next_prob[last_word]:
        if edit_distance(word, target_word) == 1:
            candidates_probs.append((word, prob))
    return candidates_probs


def not_in_dictionary(sentence):
    word2next_prob = get_word2next_prob_sorted()
    words = sentence.split(' ')
    corrected_sentence = words[0]
    for i in range(1, len(words)):
        last_word = words[i - 1]
        corrected = words[i]
        if corrected not in word2next_prob:
            candidates_probs = find_candidates_given_word(word2next_prob, last_word, words[i])
            if len(candidates_probs) != 0:
                corrected, _ = candidates_probs[0]
                print("\t{} --> {}".format(words[i], corrected))
        corrected_sentence += ' ' + corrected
    corrected_sentence = corrected_sentence.strip()
    return corrected_sentence


s = "two of thez"
s_ = not_in_dictionary(s)
print("{} --> {}".format(s, s_))


# In[37]:



def get_word2next2prob():
    all_tokens = get_tokens_from_all_corpus()
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens)

    scored = finder.score_ngrams(bigram_measures.raw_freq)
    word2next_prob = dict()
    for w1w2, prob in scored:
        w1, w2 = w1w2
        if w1 not in word2next_prob:
            word2next_prob[w1] = dict()
        word2next_prob[w1][w2] = prob
    return word2next_prob


def find_candidates(word2next2prob, words):
    list_candidates = []
    for i in range(len(words)):
        candidates = [words[i]]
        for word in word2next2prob:
            if edit_distance(word, words[i]) == 1:
                candidates.append(word)
        list_candidates.append(candidates)
    return list_candidates


import itertools
import sys
import math


def in_dictionary(sentence):
    word2next2prob = get_word2next2prob()
    words = sentence.split(' ')
    list_candidates = find_candidates(word2next2prob, words)
    print(list_candidates)

    num_tokens = float(len(word2next2prob))
    smoothing = 1.0 / num_tokens

    products = itertools.product(*list_candidates)
    max_prob = 0.0 - sys.float_info.max
    max_candi = None

    for possible_sentence in products:
        last_word = possible_sentence[0]
        sum_prob = 0.0
        for i in range(1, len(possible_sentence)):
            prob = smoothing
            cur_word = possible_sentence[i]
            if last_word in word2next2prob:
                if cur_word in word2next2prob[last_word]:
                    prob = word2next2prob[last_word][cur_word]
            prob = math.log(prob)
            sum_prob += prob
            last_word = cur_word
        print("\t{}: {}".format(possible_sentence, sum_prob))

        if prob > max_prob:
            max_prob = prob
            max_candi = possible_sentence
    return max_candi


print(in_dictionary(s))

