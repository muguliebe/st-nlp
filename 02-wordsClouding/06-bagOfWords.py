#-*- coding: utf-8 -*-
import os
from konlpy.tag import Mecab

mecab = Mecab()

class Vocaburary:
    vocab = []
    def __init__(self,filename):
        with open(filename) as f:
            pass
        ## TODO: 파일/폴더를 읽어서 vocab에 저장

    def getID(self,term):
        return self.vocab.index(term)

    def getString(self,id):
        return self.vocab[id]

vocab = []
def create_vocab(folder_name):
    pass

def bow_sequence(v, file_name):
    sequence = []
    return sequence

def bow_term_frequence(v, file_name):
    dict = {}
    return dict

if __name__ == '__main__':
    v = create_vocab("data")
    print bow_sequence(v,"data/17670_N1.txt")
    print bow_term_frequence(v,"data/17670_N1.txt")
