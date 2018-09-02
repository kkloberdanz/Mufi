#!/usr/bin/env python3

license = '''
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
'''

depth_counter = 0

import traceback
import copy
import sys
import os
from optparse import OptionParser

sys.setrecursionlimit(15000)

_toks_dct = {
    '('  : 'L_PAREN',
    ')'  : 'R_PAREN',
    '+'  : 'PLUS',
    '+.' : 'FLOAT_PLUS',
    '-'  : 'MINUS',
    '-.' : 'FLOAT_MINUS',
    '*'  : 'TIMES',
    '*.' : 'FLOAT_TIMES',
    '/'  : 'DIV',
    '/.' : 'FLOAT_DIV',
    'if' : 'IF',
    '<'  : 'LESS_THAN',
    '<=' : 'LESS_THAN_EQUAL',
    '<.' : 'FLOAT_LESS_THAN',
    '>'  : 'GREATER_THAN',
    '>=' : 'GREATER_THAN_EQUAL',
    '>.' : 'FLOAT_GREATER_THAN',
    '='  : 'EQUAL',
    '=.' : 'FLOAT_EQUAL',
    ','  : 'COMMA',
    'def': 'DEF',
    'list' : 'LIST',
    'defun': 'DEFUN',
    '__len__': 'LEN',
    'lambda': 'LAMBDA',
}

builtins = {
    '+',
    '+.',
    '-',
    '-.',
    '*',
    '*.',
    '/',
    '/.',
    '>',
    '>=',
    '>.',
    '<',
    '<=',
    '<.',
    '=',
    '=.',
    'mod',
    'def',
    'defun',
    'append',
    'list',
    'index',
    'take',
    'cut',
    '__len__',
    'cat',
    'and',
    'or',
    'exp',
}

class LispMissingStdLib(Exception):
    pass

class LispSyntaxError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LispMissingSymbolError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LispTypeError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class Token:
    def __init__(self, string):
        self.string = string
        self.lisp_type = self.getType()

    def __hash__(self):
        return hash(self.__str__())

    def _isFloat(self):
        if '.' in self.string:
            s = self.string.split('.')
            if len(s) == 2:
                a, b = s
                if a.isdigit() and b.isdigit():
                    return True
        return False

    def getType(self):
        if self.string.isdigit():
            return 'NUMBER'
        elif self._isFloat():
            return 'FLOAT'
        elif self.string.startswith('"') and self.string.endswith('"'):
            return 'STRING'
        return _toks_dct.get(self.string, 'IDENTIFIER')

    def isAtomic(self):
        return (self.lisp_type == 'NUMBER') or \
               (self.lisp_type == 'FLOAT') or \
               (self.string in builtins)  or \
               (self.lisp_type == 'IDENTIFIER')  or \
               (self.lisp_type == 'NIL')  or \
               (self.lisp_type == 'STRING')

    def __str__(self):
        return "%s: %s" % (self.string, self.lisp_type)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.lisp_type == other.lisp_type

class Lexer:
    def __init__(self, string):
        self.tokens = []
        self.toks_set = set(_toks_dct.keys())
        self.delim_set = set([' ', '\t', '\n'])
        self.string = string.strip()
        self.len_string = len(self.string)
        self.index = 0

    def addToken(self, token):
        if token:
            self.tokens.append(Token(token))

    def getChar(self):
        ret = self.string[self.index]
        self.index += 1
        return ret

    def ungetChar(self):
        self.index -= 1
        ret = self.string[self.index]
        return ret

    def hasChar(self):
        return self.index < self.len_string

    def tokenize(self):
        if self.string.count('(') != self.string.count(')'):
            raise LispSyntaxError("Mismatched parens")
        token = ''
        while self.hasChar():
            c = self.getChar()
            if c in ('(', ')'):
                if token:
                    self.addToken(token)
                    token = ''
                self.addToken(c)
                continue
            if c in self.delim_set:
                if token:
                    self.addToken(token)
                    token = ''
                continue

            if c != ' ':
                token += c
            if token in self.toks_set and self.hasChar():
                while self.hasChar() and token in self.toks_set or token in [t[:len(token)] for t in _toks_dct.keys()]:
                    token += self.getChar()
                self.ungetChar()
                token = token[:-1]
                self.addToken(token)
                token = ''

        if token:
            self.addToken(token)
        return self.tokens

    def __repr__(self):
        s = ''
        indent_level = 0
        delim = '    '
        for tok in self.tokens:

            if tok.lisp_type == 'L_PAREN':
                s += '\n' + indent_level * delim + str(tok) + '\n'
                indent_level += 1

            elif tok.lisp_type == 'R_PAREN':
                indent_level -= 1
                s += indent_level * delim + str(tok) + '\n\n'

            else:
                s += indent_level * delim + str(tok) + '\n'

        return s

def convert_bool(bool_string):
    if bool_string == 'False':
        return False
    elif bool_string == 'True':
        return True
    else:
        raise Exception("Cannot convert '{}' to bool".format(bool_string))

operations_dct = {
    '+'   : lambda x, y: LispObject(x + y, 'NUMBER'),
    '+.'  : lambda x, y: LispObject(x + y, 'FLOAT'),
    '-'   : lambda x, y: LispObject(x - y, 'NUMBER'),
    '-.'  : lambda x, y: LispObject(x - y, 'FLOAT'),
    '*'   : lambda x, y: LispObject(x * y, 'NUMBER'),
    '*.'  : lambda x, y: LispObject(x * y, 'FLOAT'),
    '/'   : lambda x, y: LispObject(x // y, 'NUMBER'),
    '/.'  : lambda x, y: LispObject(x / y, 'FLOAT'),
    'mod' : lambda x, y: LispObject(x % y, 'NUMBER'),
    'exp' : lambda x, y: LispObject(x ** y, 'FLOAT'),
    '='   : lambda x, y: LispObject(str(x == y), 'BOOL'),
    '=.'  : lambda x, y: LispObject(str(x == y), 'BOOL'),
    '>'   : lambda x, y: LispObject(str(x > y),  'BOOL'),
    '>='  : lambda x, y: LispObject(str(x >= y), 'BOOL'),
    '>.'  : lambda x, y: LispObject(str(x > y),  'BOOL'),
    '<'   : lambda x, y: LispObject(str(x < y),  'BOOL'),
    '<='  : lambda x, y: LispObject(str(x <= y), 'BOOL'),
    '<.'  : lambda x, y: LispObject(str(x < y),  'BOOL'),
    'and' : lambda x, y: LispObject(str(convert_bool(x) and convert_bool(y)), 'BOOL'),
    'or'  : lambda x, y: LispObject(str(convert_bool(x) or convert_bool(y)), 'BOOL'),
    'append' : lambda x, l: LispObject(l.value + [x], 'LIST'),
    'index' : lambda x, l: l.value[x.value],
    'take' : lambda x, l: LispObject(l.value[:x.value], 'LIST'),
    'cut' : lambda x, l: LispObject(l.value[x.value:], 'LIST'),
    '__len__' : lambda l: len(l),
    'cat' : lambda l1, l2: LispObject(l1.value + l2.value, 'LIST'),
}

class LispObject:
    def __init__(self, string, lisp_type):
        self.string = string
        self.lisp_type = lisp_type
        self.type_dct = {
            'NUMBER' : int,
            'FLOAT'  : float,
            'STRING' : str,
            'BOOL'   : bool,
            'LIST'   : list,
        }
        self.valid_ops_dct = {
            'NUMBER' : set(['=', '>', '<', '<=', '>=', '+', '-', '*', '/', 'mod', 'exp']),
            'FLOAT'  : set(['=', '>', '<', '<=', '>=', '+', '-', '*', '/', '=.', '>.', '<.', '+.', '-.', '*.', '/.', 'exp']),
            'STRING' : set(['||']),
            'LIST'   : set(['index', 'append']),
            'BOOL'   : set(['and', 'or', '=']),
        }

        self.value = self._getValue()

    def _getValue(self):
        if self.lisp_type == 'BOOL':
            if self.string == 'False':
                return 'False'
            elif self.string == 'True':
                return 'True'
            else:
                raise LispSyntaxError('Bool must be either True or False, got {}'.format(self))
        elif self.lisp_type in ['FUNCTION', 'LAMBDA']:
            return self.string
        else:
            res = self.type_dct[self.lisp_type](self.string)
            return res

    def __str__(self):
        return "{}: {}".format(self.string, self.lisp_type)

    def __repr__(self):
        return self.__str__()

    def performOperation(self, op, other):
        if other.lisp_type in ['LIST', 'STRING']:
            return operations_dct[op.string](self, other)
        else:
            if op.string not in self.valid_ops_dct[self.lisp_type] or op.string not in self.valid_ops_dct[other.lisp_type]:
                raise LispTypeError('unsupported type for op: {}, (Given: {}, {})'.format(op, self, other))

            if op.string not in operations_dct:
                raise LispSyntaxError('{} is not a supported operator!'.format(op))
            res = operations_dct[op.string](self.value, other.value)
            return res

class AST:
    def __init__(self, tree):
        self.tree = tree

    def __str__(self):
        s = ''
        for c in str(self.tree):
            if c == ']':
                s += c + '\n'
            else:
                s += c
        return s

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return (tree for tree in self.tree)

    def evaluate(self, env={}):
        for t in self.tree:
            ret = self._evaluate(t, env)
        return ret

    def _evaluateLambda(self, expr, env):
        if isinstance(expr, LispObject):
            return expr, env
        else:
            _, params, body = expr
            return LispObject((params, body), 'LAMBDA'), env

    def _evalFunction(self, args, func, env):
        if isinstance(func, LispObject):
            return self._evalAtom(func, env), env
        if len(func) != 2:
            raise LispSyntaxError("Functions must be of form ()")
        params, expr = func
        if len(params) != len(args):
            raise LispSyntaxError("Unmatched arguments, expecting {} got {}".format(params, args))

        local_env = copy.copy(env)
        for param, arg in zip(params, args):
            val, _ = self._evaluate(arg, env)
            if isinstance(val, list):
                val, env = self._evaluate(val, env)
            if val.lisp_type == 'FUNCTION':
                local_env[param.string] = env[val.string]
            elif val.lisp_type == 'LAMBDA':
                local_env[param.string] = val.string
            else:
                local_env[param.string] = val
        return self._evaluate(expr, local_env)[0], env

    def _evalAtom(self, expr, env):
        if expr.lisp_type == 'NUMBER':
            return LispObject(expr.string, 'NUMBER')

        elif expr.lisp_type == 'FLOAT':
            return LispObject(expr.string, 'FLOAT')

        elif expr.lisp_type == 'BOOL':
            return LispObject(expr.string, 'BOOL')

        elif expr.lisp_type == 'STRING':
            return LispObject(expr.string.strip('"'), 'STRING')

        elif expr.lisp_type == 'LIST':
            return LispObject(expr.string, 'LIST')

        elif expr.lisp_type == 'IDENTIFIER':
            if expr.string in env:
                # function
                if isinstance(env[expr.string], tuple):
                    return LispObject(expr.string, 'FUNCTION')
                # variable
                else:
                    return self._evalAtom(env[expr.string], env)
            else:
                raise LispMissingSymbolError("Runtime Error, Undefined symbol {}".format(expr))

        elif expr.lisp_type == 'NIL':
            return LispObject('NIL', 'NIL')

        elif expr.lisp_type == 'LAMBDA':
            return self._evaluateLambda(expr, env)[0]

        else:
            raise ValueError("Runtime Error, invalid type {}".format(expr))

    def _define_var(self, expr, env):
        if len(expr) != 3:
            raise LispSyntaxError("DEF statement must be of form (def VARIABLE VALUE), got {}".format(expr))
        _, name, value = expr
        env[name.string], _ = self._evaluate(value, env)
        return value, env

    def _define_func(self, expr, env):
        if len(expr) != 4:
            raise LispSyntaxError("DEFUN statement must be of form (defun FUNCTION_NAME ARGUMENTS EXPRESSION), got {}".format(expr))
        _, name, args, value = expr
        env[name.string] = (args, value)
        return value, env

    def _evaluateIf(self, expr, env):
        # (if BOOL T_EXPR F_EXPR)
        if len(expr) != 4:
            raise LispSyntaxError("If expression expecting: {}, got {}".format('(if BOOL T_EXPR F_EXPR)', expr))
        _, bool_expr, t_expr, f_expr = expr
        val, env = self._evaluate(bool_expr, env)
        if val.value == 'True':
            return self._evaluate(t_expr, env)
        elif val.value == 'False':
            return self._evaluate(f_expr, env)
        else:
            raise LispSyntaxError("Type BOOL must be either True or False, got {}".format(val))

    def reduce(self, op, l):
        if len(l) > 1:
            a = l[0]
            res = a.performOperation(op, self.reduce(op, l[1:]))
            return res
        elif len(l) == 1:
            return l[0]
        else:
            raise Exception('Wat? interpreter reduce is broken')

    def _evaluateList(self, expr, env):
        l = []
        for sub_expr in expr[1:]:
            value, env = self._evaluate(sub_expr, env)
            l.append(value)
        obj = LispObject(l, 'LIST')
        return obj, env

    def _evaluate(self, expr, env):
        # takes tokens as input

        global depth_counter
        depth_counter += 1

        if isinstance(expr, list):
            op = expr[0]
            if op.lisp_type == 'DEF':
                value, env = self._define_var(expr, env)
                depth_counter -= 1
                return value, env

            elif op.lisp_type == 'DEFUN':
                value, env = self._define_func(expr, env)
                depth_counter -= 1
                return value, env

            elif op in env or op.string in env:
                depth_counter -= 1
                return self._evalFunction(expr[1:], env[op.string], env)

            elif op.lisp_type == 'IF':
                depth_counter -= 1
                return self._evaluateIf(expr, env)

            elif op.lisp_type == 'LIST':
                depth_counter -= 1
                return self._evaluateList(expr, env)

            elif op.lisp_type == 'LAMBDA':
                depth_counter -= 1
                return self._evaluateLambda(expr, env)

            elif op.lisp_type in ('IDENTIFIER', 'NUMBER', 'NIL') and len(expr) == 1:
                depth_counter -= 1
                return self._evalAtom(op, env), env

            elif op.string not in builtins:
                raise LispSyntaxError("Expecting an operator, got {}".format(op))

            elif len(expr[1:]) < 2 and op.lisp_type not in ('LEN', 'INT', 'FLOAT'):
                    raise LispSyntaxError("Runtime Error, expecting at least 2 arguments for OP: {}".format(op))

            elif op.lisp_type == 'LEN':
                depth_counter -= 1
                return LispObject(len(self._evaluate(expr[1], env)[0].string), 'NUMBER'), env

            else:
                l = [self._evaluate(ex, env)[0] for ex in expr[1:]]
                depth_counter -= 1
                return self.reduce(op, l), env
        else:
            depth_counter -= 1
            return self._evalAtom(expr, env), env

class Parser:
    def __init__(self, env):
        self.env = env

    def parse(self, tokens):
        tree = []
        try:
            while tokens and tokens != [Token(')')]:
                tmp, tokens = self.exp(tokens, [])
                while tokens and tokens[0] == Token(')'):
                    tokens = tokens[1:]
                tree.append(tmp)
            ast = AST(tree)
        except IndexError as ie:
            raise LispSyntaxError("Syntax error")
        return ast

    def match(self, expected, given):
        if expected != given:
            raise LispSyntaxError("Error, expected: {}, given: {}".format(expected, given))

    def headTail(self, tokens):
        head = tokens[0]
        tail = tokens[1:]
        return head, tail

    def getArgs(self, tokens):
        l = []
        tok, tokens = self.headTail(tokens)
        self.match(Token("("), tok)
        while True:
            tok, tokens = self.headTail(tokens)
            if tok.lisp_type == 'R_PAREN':
                break
            elif tok.lisp_type != 'IDENTIFIER':
                raise LispSyntaxError("Expecting arguments, got {}".format(tok))
            else:
                l.append(tok)

        self.match(Token(")"), tok)
        return l, tokens

    def getEndParen(self, tokens):
        paren_num = 1
        for i, tok in enumerate(tokens):
            if tok == Token('('):
                paren_num += 1

            elif tok == Token(')'):
                paren_num -= 1

            if paren_num == 0:
                return i
        return -1

    def funcDef(self, tokens, l):
        # (defun f (args) (expr))
        func_name, tokens = self.headTail(tokens)
        self.env[func_name] = True
        args, tokens = self.getArgs(tokens)
        end_paren_index = self.getEndParen(tokens)
        r_paren = tokens[end_paren_index]
        func_tokens = tokens[:end_paren_index]
        self.match(Token(')'), r_paren)
        expr, _ = self.exp(func_tokens, [])
        tokens = tokens[end_paren_index + 1:]
        return [func_name, args, expr], tokens

    def exp(self, tokens, l):
        tok, tokens = self.headTail(tokens)
        if tok.isAtomic():
            l.append(tok)
            return l, tokens
        self.match(Token("("), tok)

        op, tokens = self.headTail(tokens)

        if op.lisp_type == 'DEFUN':
            l, tokens = self.funcDef(tokens, l)
            return [op] + l, tokens

        l.append(op)
        tok, tokens = self.headTail(tokens)
        while tok.isAtomic():
            l.append(tok)
            tok, tokens = self.headTail(tokens)

        if tok.string == ')':
            return l, [tok] + tokens

        elif tok.string == '(':
            while tok.string == '(':
                tmp, tokens = self.exp([tok] + tokens, [])
                l.append(tmp)
                tok, tokens = self.headTail(tokens)
                self.match(Token(')'), tok)

                if tokens:
                    tok, tokens = self.headTail(tokens)
                    while tok.isAtomic():
                        l.append(tok)
                        tok, tokens = self.headTail(tokens)
                else:
                    tok, tokens = self.headTail(tokens)
                    self.match(Token(')'), tok)
                    break

            return l, [tok] + tokens

        else:
            raise LispSyntaxError("Expecting ')' or '(', got {}".format(tok))

class Compiler:

    op_mapping = {
        '+': 'ADD',
        '-': 'SUB',
        '*': 'MUL',
        '/': 'DIV',
    }

    def compile(self, ast):
        for tree in ast:
            self._compile(tree)
        print('HALT')

    def _compile(self, tree):
        if isinstance(tree, list):
            op = tree[0]
            tail = tree[1:]
            for sub_tree in tail[::-1]:
                self._compile(sub_tree)
            num_operations = len(tree[1:]) - 1
            op_translation = self.op_mapping[op.string]
            print('\n'.join([op_translation] * num_operations))
        else:
            print('PUSH')
            print(tree.string)

def interpret(user_input, env, parser=None, compiler=False):
    err = False
    if parser is None:
        parser = Parser(env)
    try:
        lex = Lexer(user_input)
        tokens = lex.tokenize()
        ast = parser.parse(tokens)
        if compiler:
            Compiler().compile(ast)

        (value, env) = ast.evaluate(env)
        return value, env, err

    except IndexError as ie:
        msg = 'error: list out of range'
        err = True

    except EOFError as eof:
        msg = 'exiting'
        err = True

    except LispSyntaxError as lse:
        msg = 'syntax error: {}'.format(lse)
        err = True

    except LispMissingSymbolError as lmse:
        msg = 'Missing symbol: {}'.format(lmse)
        err = True

    except LispTypeError as lte:
        msg = 'Type error: {}'.format(lte)
        err = True

    except Exception as e:
        traceback.print_exc()
        msg = 'Internal error (This should never happen. Please submit a bug report with this error!): {}'.format(e)
        err = True
    return msg, env, err

def loadFile(filename, env={}):
    with open(filename, 'r') as f:
        parser = Parser(env)
        lines = []
        for line in f:
            precomment = line.split(';')[0]
            lines.append(precomment)
        value, env, err = interpret('\n'.join(lines), env, parser)
    return value, env, err

def shell(env={}, quiet=False, compiler=False):
    if not quiet:
        print(license)
        print('q to exit')
        prompt = '> '
    else:
        prompt = ''
    parser = Parser(env)
    while True:
        try:
            parser.env = env

            user_input = input(prompt) + ' '

            while user_input != '' and user_input.count('(') > user_input.count(')'):
                user_input += input() + ' '

            if user_input.strip() == 'q':
                return

            elif not user_input or user_input == ' ':
                continue

            value, env, err = interpret(user_input, env, parser, compiler)
            print(value)

        except KeyboardInterrupt as ki:
            print('<CANCELLED>')
            continue

        except EOFError as eof:
            return

color_dct = {
    'green': '\033[92m',
    'red'  : '\033[91m',
    'end'  : '\033[0m',
}

def color(_color, string):
    return color_dct.get(_color, '') + string + color_dct['end']

def _assertEqual(result, expected):
    passing = str(result) == str(expected)
    if passing:
        string = color("green", "PASS")# +  " {}".format(result)
    else:
        string = color("red", "FAIL")# + " Result '{}' -> Expecting: '{}'".format(result, expected)
    return string, passing

def test(env={}):
    test_cases = [
    # (Test, env, expected result)
        ('(+ 1 2)',         {},              LispObject(3, 'NUMBER')),
        ('(+ x x)',         {'x':LispObject(3, 'NUMBER')},         LispObject(6, 'NUMBER')),
        ('(+ x x 4)',       {'x':LispObject(3, 'NUMBER')},         LispObject(10, 'NUMBER')),
        ('(+ (* 8 7) (- 4 (+ 3 2)) (+ (- 9 1) 4 (* 8 9)))', {'x':5}, LispObject(139, 'NUMBER')),
        ('(if (> 1 x) x 4)',{'x':LispObject(3, 'NUMBER')},         Token('4')),
        ('(if (< 1 x) x 4)',{'x':LispObject(3, 'NUMBER')},         Token('3')),
        ('(def x 4)',       {},              Token('4')),
        ('(defun f (x y) (+ x y))', {},      [Token('+'), Token('x'), Token('y')]),
        ('(= x x)',         {'x':LispObject(5, 'NUMBER')},         LispObject('True', 'BOOL')),
        ('(+. 3.3 4.5)',     {},              LispObject('7.8', 'FLOAT')),
        ('(+. 3.3 y)',       {'y':LispObject(4.5, 'FLOAT')},       LispObject('7.8', 'FLOAT')),
        ('(def y 17) (= (gcd (square y) y) y)', env, LispObject('True', 'BOOL')),
        ('(exp (factorial 3) 5)', env, LispObject('7776', 'FLOAT')),
        ('(defun add3 (x) (+ x 3)) (add3 5)', {}, LispObject('8', 'NUMBER')),
        ('(append 1 (list 1 2 3 4))', {}, "[1: NUMBER, 2: NUMBER, 3: NUMBER, 4: NUMBER, 1: NUMBER]: LIST"),
        ('(index 2 (list 11 22 33 44 55))', {}, LispObject(33, 'NUMBER')),
        ('(def l (list 11 22 33 44 55)) (index 2 l)', {}, LispObject(33, 'NUMBER')),
        ('(def l (list 1 2 3 4 5 6 7)) (take 5 l)', {}, "[1: NUMBER, 2: NUMBER, 3: NUMBER, 4: NUMBER, 5: NUMBER]: LIST"),
        ('(def l (list 1 2 3 4 5 6 7)) (cut 5 l)', {}, "[6: NUMBER, 7: NUMBER]: LIST"),
        ('(append 5 (append 6 (list 1 2 3 4)))', {}, "[1: NUMBER, 2: NUMBER, 3: NUMBER, 4: NUMBER, 6: NUMBER, 5: NUMBER]: LIST"),
        ('(def l (list 1 2 3 4)) (append 6 (append 7 (l)))', {}, "[1: NUMBER, 2: NUMBER, 3: NUMBER, 4: NUMBER, 7: NUMBER, 6: NUMBER]: LIST"),
        ('(def l (list 1 2 3 4 5 6 7)) (last l)', env, LispObject('7', 'NUMBER')),
        ('(append 1 (append 2 (list)))', {}, "[2: NUMBER, 1: NUMBER]: LIST"),
        ('(range 0 10)', env, "[0: NUMBER, 1: NUMBER, 2: NUMBER, 3: NUMBER, 4: NUMBER, 5: NUMBER, 6: NUMBER, 7: NUMBER, 8: NUMBER, 9: NUMBER]: LIST"),
        ('(map factorial (range 0 10))', env, "[1: NUMBER, 1: NUMBER, 2: NUMBER, 6: NUMBER, 24: NUMBER, 120: NUMBER, 720: NUMBER, 5040: NUMBER, 40320: NUMBER, 362880: NUMBER]: LIST"),
        ('(filter even (range 0 10))', env, "[0: NUMBER, 2: NUMBER, 4: NUMBER, 6: NUMBER, 8: NUMBER]: LIST"),
        ('(filter (lambda (x) (not (even x))) (range 0 10))', env, "[1: NUMBER, 3: NUMBER, 5: NUMBER, 7: NUMBER, 9: NUMBER]: LIST"),
        ('(cat (range 0 5) (range 10 15))', env, "[0: NUMBER, 1: NUMBER, 2: NUMBER, 3: NUMBER, 4: NUMBER, 10: NUMBER, 11: NUMBER, 12: NUMBER, 13: NUMBER, 14: NUMBER]: LIST"),
        ('(reduce add 0 (range 0 10))', env, "45: NUMBER"),
        ('(reduce mul 1 (range 1 10))', env, "362880: NUMBER"),
        ('(sum (filter even (range 0 10)))', env, "20: NUMBER"),
        ('(product (filter even (range 1 10)))', env, "384: NUMBER"),
        ('(map sum (list (range 0 10) (range 0 20)))', env, "[45: NUMBER, 190: NUMBER]: LIST"),
        ('(map sum (list (list 1 2) (list 3 4) (list 5 6)))', env, "[3: NUMBER, 7: NUMBER, 11: NUMBER]: LIST"),
        ('(zip (range 0 5) (range 100 105))', env, "[[0: NUMBER, 100: NUMBER]: LIST, [1: NUMBER, 101: NUMBER]: LIST, [2: NUMBER, 102: NUMBER]: LIST, [3: NUMBER, 103: NUMBER]: LIST, [4: NUMBER, 104: NUMBER]: LIST]: LIST"),
        ('(map sum (zip (range 0 5) (range 100 105)))', env, "[100: NUMBER, 102: NUMBER, 104: NUMBER, 106: NUMBER, 108: NUMBER]: LIST"),
        ('(reduce mul 1 (map sum (zip (list 1 2 3) (list 4 5 6))))', env, "315: NUMBER"),
        ('(map length (list (range 0 10) (range 4 5) (range 0 4) (zip (range 0 10) (range 0 13))))', env, "[10: NUMBER, 1: NUMBER, 4: NUMBER, 10: NUMBER]: LIST"),
        ('(map length (zip (range 0 10) (range 100 200)))', env, "[2: NUMBER, 2: NUMBER, 2: NUMBER, 2: NUMBER, 2: NUMBER, 2: NUMBER, 2: NUMBER, 2: NUMBER, 2: NUMBER, 2: NUMBER]: LIST"),
        ('(collatz 13)', env, "[1: NUMBER, 2: NUMBER, 4: NUMBER, 8: NUMBER, 16: NUMBER, 5: NUMBER, 10: NUMBER, 20: NUMBER, 40: NUMBER, 13: NUMBER]: LIST"),
        ('(and False True)', env, "False: BOOL"),
        ('(and False False)', env, "False: BOOL"),
        ('(and True False)', env, "False: BOOL"),
        ('(and True True)', env, "True: BOOL"),
        ('(or True True)', env, "True: BOOL"),
        ('(or False True)', env, "True: BOOL"),
        ('(or True False)', env, "True: BOOL"),
        ('(or False False)', env, "False: BOOL"),
        ('(and True False True)', env, "False: BOOL"),
        ('(or True False True)', env, "True: BOOL"),
        ('(max (list 1 2 6 4 2 5))', env, "6: NUMBER"),
        ('(min (list 1 2 6 4 2 5))', env, "1: NUMBER"),
        ('(map (lambda (x) (+ 1 x)) (list 1 2 3))', env, "[2: NUMBER, 3: NUMBER, 4: NUMBER]: LIST"),
        ('(quicksort (list 1 1 3 4 3 5 1))', env, "[1: NUMBER, 1: NUMBER, 1: NUMBER, 3: NUMBER, 3: NUMBER, 4: NUMBER, 5: NUMBER]: LIST"),
        ('(quicksort (list 1 4 6 2 0 3 4 1 9))', env, "[0: NUMBER, 1: NUMBER, 1: NUMBER, 2: NUMBER, 3: NUMBER, 4: NUMBER, 4: NUMBER, 6: NUMBER, 9: NUMBER]: LIST"),
        ('(fibonacci 10)', env, "[1: NUMBER, 1: NUMBER, 2: NUMBER, 3: NUMBER, 5: NUMBER, 8: NUMBER, 13: NUMBER, 21: NUMBER, 34: NUMBER, 55: NUMBER, 89: NUMBER]: LIST"),
        ('(sum (filter even (filter (lambda (x) (< x 4000000)) (fibonacci 1000))))', env, "4613732: NUMBER"), # https://projecteuler.net/problem=2
        ('(filter isprime (range (neg 5) 20))', env, "[2: NUMBER, 3: NUMBER, 5: NUMBER, 7: NUMBER, 11: NUMBER, 13: NUMBER, 17: NUMBER, 19: NUMBER]: LIST"),
        ('(def l (list 1 2 3 4 5)) (sum l) (sum l)', env, "15: NUMBER"),
    ]

    all_pass = True
    pass_casses = 0
    for i, test_tpl in enumerate(test_cases):
        testcase, env, expected = test_tpl
        result, env, err = interpret(testcase, env, Parser(env))
        string, passing = _assertEqual(result, expected)

        msg = "{:>4} {} Input: '{}' -> '{}'".format(i+1, string, testcase, result)
        if not passing:
            msg += ", Expecting: '{}'".format(expected)
        else:
            pass_casses += 1

        print(msg)
        all_pass = passing and all_pass

    print('\n')
    if all_pass:
        print(color('green', 'ALL PASSING, {}/{} Passed'.format(pass_casses, i+1)))
    else:
        print(color('red', 'ERRORS ENCOUNTERED, {}/{} Passed'.format(pass_casses, i+1)))
    return all_pass


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--quiet', action='store_true', default=False, help='Quietly launch shell')
    parser.add_option('--test', action='store_true', default=False, help='Run unit tests')
    parser.add_option('--compiler',
                      action='store_true',
                      default=False,
                      help='run bytecode compiler')
    options, args = parser.parse_args()

    if options.compiler:
        shell(compiler=True)
        sys.exit(0)

    try:
        value, env, err = loadFile(os.getenv('MUFI_STDLIB', 'stdlib.lisp'))
    except FileNotFoundError as e:
        raise LispMissingStdLib('Try setting the environment variable MUFI_STDLIB to point the full path to stdlib.lisp')

    if options.test:
        test(env)

    elif args:
        filename = args[0]
        value, env, err = loadFile(filename, env)
        if err:
            print("Error: {}".format(value))
        else:
            shell(env=env, quiet=options.quiet)
    else:
        shell(env=env, quiet=options.quiet)
