#-*- coding: utf-8 -*-
from konlpy.tag import Mecab

mecab = Mecab()
result = mecab.pos(u'SK텔레콤은 이번 전시에서 16년부터 에릭슨 인텔과 공동 개발한 5G 이동형 인프라차량을 처음 선보인다')
for r in result:
    val = r[0].encode('utf-8')
    tag = r[1].encode('utf-8')
    print val + "/" + tag