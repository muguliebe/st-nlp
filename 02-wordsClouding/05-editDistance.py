#-*- coding: utf-8 -*-
KO_INIT_S = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ',
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
] ## 19

KO_INIT_M =[
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ',
    'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
] ## 21

KO_INIT_E = [
    0, 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ',
    'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
] ## 28

HAN_BEGIN = 44032 ## u'가'
HAN_END = 55203 ## u'힣'
CHOSUNG = 588
JUNGSUNG = 28

def toKoJaso(text):
    if text == '' or len(text) == 0: return None
    jaso = ''
    result = []
    list_text = list(text)
    for c in list_text:
        char_code = ord(c)
        if (char_code >= HAN_BEGIN and char_code <= HAN_END):
            #print char_code, c
            ce = char_code - HAN_BEGIN
            result.append(KO_INIT_S[ce/CHOSUNG])
            #print KO_INIT_S[ce/CHOSUNG]

            ce = ce % CHOSUNG
            #print KO_INIT_M[ce/JUNGSUNG]
            result.append(KO_INIT_M[ce/JUNGSUNG])
            ce = ce % JUNGSUNG
            if (ce != 0):
                result.append(KO_INIT_E[ce])
                #print KO_INIT_E[ce]
        else:
            pass
            #print 'Not Hangul'

    jaso =  "".join(result)
    return jaso

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# jaso1 = toKoJaso(u"박진영")
# jaso2 = toKoJaso(u"박진혁")
# print jaso1, jaso2
# d = edit_distance(u"박진영", u"박진혁")
# d2 = edit_distance(jaso1, jaso2)
# print d, d2

candidates = ["쟈니얀", "쟈니욘", "쟈니양"]

jasoorigin = toKoJaso(u'쟈니윤')
for idx, val in enumerate(candidates):
    jasocompare = toKoJaso(val.decode('utf-8'))
    d = edit_distance(jasoorigin.decode('utf-8'), jasocompare.decode('utf-8'))
    print val, d, jasocompare