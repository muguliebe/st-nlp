from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk


def read_snippet():
    s = ""
    with open("data/input.txt", encoding="utf-8") as f:
        s = f.read()
    return s


def tokenizer(snippet):
    tokens = None
    ################################
    ToDo = "snippet --> tokens"
    ################################
    snippet = snippet.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(snippet)

    return tokens


def stemming(tokens):
    porter_stemmed = None
    ################################
    ToDo = "Use Porter stemmer"
    ################################
    porter = PorterStemmer()
    porter_stemmed = [porter.stem(token) for token in tokens]

    return porter_stemmed


def lematizer(tokens):
    nltk.download('wordnet')
    ################################
    ToDo = "Use WordNet lemmatizer"
    ################################
    wordnet_lemmas = None
    wordnet = WordNetLemmatizer()
    wordnet_lemmas = [wordnet.lemmatize(token) for token in tokens]

    return wordnet_lemmas


def porter_vs_wordnet():
    tokens = ['generate', 'generates', 'generated', 'generating', 'general', 'generally', 'generic', 'generically',
              'generous', 'generously']
    ################################
    ToDo = "Compare porter stemming result and wordnet lematizer result"
    ################################
    porter = PorterStemmer()
    porter_stemmed = [porter.stem(token) for token in tokens]

    wordnet = WordNetLemmatizer()
    wordnet_lemmas = [wordnet.lemmatize(token) for token in tokens]
    print(porter_stemmed)
    print()
    print(wordnet_lemmas)


def show_stopwords():
    stop = stopwords.words('english')
    print("Stopword list: {}".format(stop))


def check_stopwords(tokens):
    removed_stopwords = None
    ################################
    ToDo = "tokens --> stopwords removed tokens"
    ################################
    removed_stop = [t for t in wordnet_lemmas if t not in stop]

    return removed_stopwords


def test_edit_distance(s1, s2):
    dist = 0
    ################################
    ToDo = "Use edit distance"
    ################################
    dist = edit_distance(s1, s2)
    return dist


def main():
    snippet = read_snippet()

    all_tokens = tokenizer(snippet)
    # stemming(all_tokens)
    # lematizer(all_tokens)
    # porter_vs_wordnet():
    # show_stopwords()
    # check_stopwords(all_tokens)

    s1 = "This is a test sentence"
    s2 = "This is aa test sentence"
    # print(test_edit_distance(s1, s2))


if __name__ == "__main__":
    main()
