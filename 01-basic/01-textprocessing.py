import numpy as np
import re

sentences = []
sentences.append('파이썬 학습 자료 공유')
sentences.append('비아그라 시알리스 무료상담')
sentences.append('NLP 공부는 재미있어요')
sentences.append('비@#$아@!그@!#라 정품')
sentences.append('정품  비아그라 추천합니다')

sentences.append('수요일 미팅 정리')
sentences.append('비아 그라 문의')
sentences.append('하나비 아그라수')
sentences.append('(광고) 비아그라 구매')
sentences.append('일정 관련 문의 답변')

list_gold = []
list_gold.append(0)
list_gold.append(1)
list_gold.append(0)
list_gold.append(1)
list_gold.append(1)

list_gold.append(0)
list_gold.append(1)
list_gold.append(0)
list_gold.append(1)
list_gold.append(0)


def evaluation(list_ours, list_gold):
    cnt_true_positive = 0.0
    cnt_true_negative = 0.0
    cnt_false_positive = 0.0
    cnt_false_negative = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    #     compare ours with ground truth
    #     blank start
    for i in range(len(list_gold)):
        if list_gold[i] == 1:
            if list_ours[i] == 1:
                cnt_true_positive += 1.0
            else:
                cnt_false_negative += 1.0
                print("\tFN: {}".format(sentences[i]))
        else:
            if list_ours[i] == 1:
                cnt_false_positive += 1.0
                print("\tFP: {}".format(sentences[i]))
            else:
                cnt_true_negative += 1.0
    if cnt_true_positive + cnt_false_positive != 0.0:
        precision = cnt_true_positive / (cnt_true_positive + cnt_false_positive)
    if cnt_true_positive + cnt_false_negative != 0.0:
        recall = cnt_true_positive / (cnt_true_positive + cnt_false_negative)
    # blank end

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
    return precision, recall, f1


def process(list_ours):
    p, r, f1 = evaluation(list_ours, list_gold)
    print("P: {0:.4f}\nR: {1:.4f}\nF1: {2:.4f}".format(p, r, f1))


def exact_filtering(list_s):
    list_labels = np.zeros((len(list_s)))
    #     blank start
    for i in range(len(list_s)):
        if "비아그라" in list_s[i]:
            list_labels[i] = 1
        else:
            list_labels[i] = 0
            #     blank end
    return list_labels


def spchar_handling(list_s):
    list_labels = np.zeros((len(list_s)))
    ################################
    ToDo = "remove special character"
    ################################
    return list_labels


def main():
    list_exact = exact_filtering(sentences)
    process(list_exact)

    # list_spchar = spchar_handling(sentences)
    # process(list_spchar)


if __name__ == "__main__":
    main()
