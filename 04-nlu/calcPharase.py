import sys

sys.path.insert(0, "../..")

tokens = ('CASHBOOK', 'ACCOUNT_NAME', 'INFORM_VERB', 'RECOMMEND_VERB',)

literals = []


def t_CASHBOOK(t):
    r'(월급|급여|관리비|적금)'
    return t


def t_ACCOUNT_NAME(t):
    r'(계좌|통장)'
    return t


def t_INFORM_VERB(t):
    r'(알려줘)'
    return t


def t_RECOMMEND_VERB(t):
    r'(추천해줘)'
    return t


t_ignore = '\t '


def t_error(t):
    print("EEROR %s" % t.value[0])
    t.lexer.skip(1)


def t_newline(t):
    r'\n'
    pass


import ply.lex as lex

lex.lex()


def p_expression_intent(p):
    """intent : ask_account
            | ask_balance
            | recommend"""
    print(p[1])


def p_expression_recommend(p):
    "recommend : account_name recommend_verb"
    p[0] = "recommend"


def p_expression_ask_account(p):
    "ask_account : cashbook account_name inform_verb"
    p[0] = "ask.account"


def p_expression_ask_balance(p):
    "ask_balance : account_name inform_verb"
    p[0] = "ask.balance"


def p_expression_cashbook(p):
    "cashbook : CASHBOOK"
    if p[1] == '스벅':
        p[0] = '스타벅스'
    p[0] = p[1]


def p_expression_account_name(p):
    "account_name : ACCOUNT_NAME"
    p[0] = p[1]


def p_expression_inform_verb(p):
    "inform_verb : INFORM_VERB"
    p[0] = p[1]


def p_expression_recommend_verb(p):
    "recommend_verb : RECOMMEND_VERB"
    p[0] = p[1]


def p_error(p):
    if p:
        print("ERROR")


import ply.yacc as yacc

yacc.yacc()

s = input('calc>')
print(yacc.parse(s))