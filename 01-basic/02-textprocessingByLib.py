# coding: utf-8

# In[54]:


snippet = """  
When the Apple Watch first came out last year, Engadget published not one but two reviews. 
There was the \"official\" review, which provided an overview of the device's features and, 
more important, attempted to explain who, if anyone, should buy it. 
Then there was a piece I wrote, focusing specifically on the watch's 
capabilities (actually, drawbacks) as a running watch. Although we knew that many readers 
would be interested in that aspect of the device, we were wary of derailing the review by 
geeking out about marathoning.

This year, we needn't worry about that. With the new Apple Watch Series 2, 
the company is explicitly positioning the device as a sports watch. 
In particular, the second generation brings a built-in GPS radio for more accurate 
distance tracking on runs, walks, hikes, bike rides and swims. 
Yes, swims: It's also waterproof this time, safe for submersion in up to 50 meters of water.

Beyond that, the other changes are performance-related, including a faster chip, 
longer battery life and a major software update that makes the watch easier to use. 
Even so, the first-gen version, which will continue to be sold at a lower price, 
is getting upgraded with the same firmware and dual-core processor. 
That means, then, that the Series 2's distinguishing features are mostly about fitness. 
And if you don't fancy yourself an athlete, we can think of an even smarter buy.  
"""

# In[75]:


from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import WhitespaceTokenizer
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# tknzr = TweetTokenizer()
s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
# twitter
# tokens = tknzr.tokenize(s0)

# regular expression
# tokens = RegexpTokenizer(r'\w+').tokenize(s0)

# word and sentence
tokens = word_tokenize(snippet)
tokens = sent_tokenize(snippet)

# splits text on whitespace and punctuation:
# tokens = wordpunct_tokenize(snippet)

# tokens = WhitespaceTokenizer().span_tokenize(snippet)

print(tokens)

# In[56]:


# tokenizer
from nltk.tokenize import RegexpTokenizer

snippet = snippet.lower()
tokens = RegexpTokenizer(r'\w+').tokenize(snippet)

print(tokens)

# In[30]:


# stemming
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
porter_stemmed = [porter.stem(token) for token in tokens]
print(porter_stemmed)

# In[76]:


# Lemmatizer
from nltk.stem.wordnet import WordNetLemmatizer

wordnet = WordNetLemmatizer()

wordnet_lemmas = [wordnet.lemmatize(token) for token in tokens]
print(wordnet_lemmas)

# In[72]:


# porter vs wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

tokens = ['generate', 'generates', 'generated', 'generating', 'general', 'generally', 'generic', 'generically',
          'generous', 'generously']
porter = PorterStemmer()
porter_stemmed = [porter.stem(token) for token in tokens]

wordnet = WordNetLemmatizer()
wordnet_lemmas = [wordnet.lemmatize(token) for token in tokens]
print(porter_stemmed)
print()
print(wordnet_lemmas)

# In[77]:


# Stopword handling

from nltk.corpus import stopwords

stop = stopwords.words('english')

# print("Stopword list: {}".format(stop))
removed_stop = [t for t in wordnet_lemmas if t not in stop]

print(removed_stop)

# In[78]:


from nltk.metrics import edit_distance

s1 = "This is test"
s2 = "Thes is test"

dist = edit_distance(s1, s2)
print(dist)

