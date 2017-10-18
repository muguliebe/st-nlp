import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')


# pos tag list
def print_tag_list():
    nltk.help.upenn_tagset()


# POS tagging demo
def pos_tagging(sentence):
    print("POS tagging")
    text = word_tokenize(sentence)
    ################################
    ToDo = "Use POS tagger"
    ################################
    print(text)


# NER demo
def NER(sentence):
    print("NER Demo")
    # http://text-processing.com/demo/tag/
    ################################
    ToDo = "Use NER"
    ################################


from nltk.corpus import treebank
from nltk.tag import hmm


def HMM():
    train_data = treebank.tagged_sents()[:3000]
    print(train_data[0])

    s1 = "Today is a good day ."
    s2 = "Joe met Joanne in Delhi ."
    s3 = "Chicago is the birthplace of Ginny"

    ################################
    ToDo = "Use HMM"
    ################################


def main():
    # print_tag_list()

    s1 = "And now for something completely different"
    pos_tagging(s1)

    s2 = "Cho and John are working at Yonsei."
    NER(s2)


if __name__ == "__main__":
    main()
