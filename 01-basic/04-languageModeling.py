# coding: utf-8

# # N-gram generation by library

# count word, make dictionary
# make N-gram
from nltk import ngrams
import nltk


def ngram_example():
    sentence = 'This is a foo bar sentences and i want to ngramize it'
    n = 2
    my_ngrams = ngrams(sentence.split(), n)
    for grams in my_ngrams:
        print(grams)


ngram_example()

# # Corpus에서 Bigram 계산하기

# In[1]:


base_folder = "./data2/"
classes = ['business', 'entertainment', 'politics', 'sport', 'tech']

# In[9]:


from nltk.collocations import BigramCollocationFinder


def get_tokens_from_all_corpus():
    all_tokens = []
    for i in range(len(classes)):
        fname = base_folder + classes[i] + '.txt'
        with open(fname, encoding='utf-8') as f:
            for line in f:
                all_tokens += nltk.word_tokenize(line.strip())
    return all_tokens


def bigram_example():
    all_tokens = get_tokens_from_all_corpus()
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(all_tokens)

    finder.apply_freq_filter(10)

    print(finder.nbest(bigram_measures.raw_freq, 10))

    scored = finder.score_ngrams(bigram_measures.raw_freq)
    print(scored)


bigram_example()


# # Library를 활용한 sentence generation

# In[157]:


def generate_sentence(start_words):
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

    last_word = start_words.split(' ')[-1]
    sentence = start_words + ' '

    bGenerating = True
    max_len = 20
    while (bGenerating):
        if len(sentence.split(' ')) > max_len:
            break
        if last_word not in word2next_prob:
            break
        next_word, _ = word2next_prob[last_word][0]
        sentence += next_word + ' '
        if next_word == '.':
            bGenerating = False
        last_word = next_word
    sentence = sentence.strip()
    return sentence


s = generate_sentence("We must")
print(s)

# # Category 별로 sentence generation

# In[5]:


import random
import nltk


def get_tokens_by_class(idx_class):
    all_tokens = []
    fname = base_folder + classes[idx_class] + '.txt'

    with open(fname, encoding="utf-8") as f:
        for line in f:
            all_tokens += nltk.word_tokenize(line.strip())
            #             all_tokens += line.strip().split(' ')
    return all_tokens


def get_biCounts(all_tokens):
    length = len(all_tokens)
    bigram_table = dict()
    for x in range(0, length - 1):
        if all_tokens[x] in bigram_table:
            if all_tokens[x + 1] in bigram_table[all_tokens[x]]:
                bigram_table[all_tokens[x]][all_tokens[x + 1]] += 1
            else:
                bigram_table[all_tokens[x]][all_tokens[x + 1]] = 1
        else:
            bigram_table[all_tokens[x]] = dict()
            bigram_table[all_tokens[x]][all_tokens[x + 1]] = 1
    return bigram_table


def get_biTable(idx_class):
    all_tokens = get_tokens_by_class(idx_class)
    bigram_table = get_biCounts(all_tokens)
    return bigram_table


def get_biSentence(min_len, max_len, table, sentence=''):
    length = len(sentence)
    if length == 0:
        sentence = choose_next(table['.'])
    sentence_tokens = nltk.word_tokenize(sentence)
    last_word = sentence_tokens[-1]
    for x in range(max_len):
        generating = True
        while (generating):
            if last_word in table:
                next_word = choose_next(table[last_word])
            else:
                next_word = random.choice(table.keys())
            generating = False
            if (next_word == '.' and length < min_len):
                generating = True
        sentence = sentence + ' ' + next_word
        if next_word == '.':
            return sentence
        length += 1
        last_word = next_word
    return sentence + '.'


def choose_next(word2cnt):
    max_cnt = -1
    next_word = ''
    for word, cnt in word2cnt.items():
        if cnt > max_cnt:
            next_word = word
            max_cnt = cnt
    return next_word


for i in range(len(classes)):
    biTable = get_biTable(i)
    s = get_biSentence(2, 50, biTable, 'We must')
    print("{}: {}".format(classes[i], s))

