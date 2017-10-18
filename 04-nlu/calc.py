import sys

sys.path.insert(0, "../..")

tokens = ('NUMBER',)

literals = ['+']


def t_NUMBER(t):
    r'[0-9]+'  # r'\d+
    t.value = int(t.value)
    return t


def t_error(t):
    print("EEROR %s" % t.value[0])
    t.lexer.skip(1)


def t_newline(t):
    r'\n'
    pass


import ply.lex as lex

lex.lex()

precedence = (('left', '+'),)


def p_statement_expr(p):
    "statement : expression"
    return (p[1])


def p_expr_binop(p):
    "expression : expression '+' expression"
    p[0] = p[1] + p[3]


def p_expr_number(p):
    "expression : NUMBER"
    p[0] = p[1]
    print(p[1])


def p_error(p):
    if p:
        print("ERROR")


import ply.yacc as yacc

yacc.yacc()

s = input('calc>')
print(yacc.parse(s))