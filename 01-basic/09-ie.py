# coding: utf-8

# In[68]:

import nltk
import re


def find_relation_in():
    IN = re.compile(r'.*\bin\b(?!\b.+ing)')

    for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
        for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=IN):
            print(nltk.sem.rtuple(rel))
            print("\t{}".format(nltk.sem.rtuple(rel, lcon=True, rcon=True)))
            #     break


# In[70]:


def find_relation_of():
    OF = re.compile(r'.*\bof\b.*')

    for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
        for rel in nltk.sem.extract_rels('PER', 'ORG', doc, corpus='ieer', pattern=OF):
            print(nltk.sem.rtuple(rel))
            print("\t{}".format(nltk.sem.rtuple(rel, lcon=True, rcon=True)))


# In[42]:


import re

IN_ = re.compile(r'.*\bin\b')
IN = re.compile(r'.*\bin\b(?!\b.+ing)')

# s = IN.findall("in moving, in the school, in company")

o = "a in moving there who in the sky"
s = IN.findall(o)
s_ = IN_.findall(o)
print(s)
print(s_)
