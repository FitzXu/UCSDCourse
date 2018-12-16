#!/usr/bin/env python3

import re
from parsec import *
from nano import *
from subprocess import run, PIPE

def fileToNano(f: str) -> Expression:
  with open(f) as fc:
    return parseNano(fc.read())

def parseNano(s: str) -> Expression:
  parse_result = run(["./nano/nanoml.byte", s], stdout=PIPE, universal_newlines=True, check=True)
  return eval(parse_result.stdout)

# ignore cases.
whitespace = regex(r'\s+', re.MULTILINE)
ignore = many((whitespace))
lexeme = lambda p: p << ignore  # skip all ignored characters.

lparen = lexeme(string('('))
rparen = lexeme(string(')'))
comma = lexeme(string(','))


# primitives: int, bool
intty = lexeme(string('int')).result(IntTy())
boolty = lexeme(string('bool')).result(BoolTy())

@generate("A free type variable")
def symbol():
  id = yield regex(r'\_[a-zA-Z0-9]*')
  return VarTy(id)


atom = intty | boolty | symbol 

@generate('An arrow')
def arrow():
    '''Parse arrow type.'''
    yield string("arrow")
    yield lparen
    l = yield ttype
    yield comma
    r = yield ttype
    yield rparen
    return ArrowTy(l, r)

@generate("A list")
def listy():
    '''Parse list type.'''
    yield string("list")
    yield lparen
    t = yield ttype
    yield rparen
    return ListTy(t)

ttype = atom | arrow | listy

program = ignore >> ttype

def parseType(s: str) -> Type:
  try:
    return program.parse(s)
  except ParseError as e:
    print("Parse error; can't parse type: %s" % s)
