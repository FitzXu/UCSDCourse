#!/usr/bin/env python3

from nano import *
from pyswip import Prolog
from subprocess import run, PIPE

from nanoParser import parseNano, parseType, fileToNano

def fileToType(f: str, with_lib=True) -> Type:
  return parseType(exprToType(fileToNano(f), with_lib=with_lib))

def exprToType(e: Expression, with_lib=True) -> str:
  p = Prolog()
  p.consult('types.pl')
  
  builtins = "[" + ",".join([
    "[null,arrow(list(X), bool)]",
    "[hd,arrow(list(Y), Y)]",
    "[tl,arrow(list(Z), list(Z))]",
    ]) + "]"
  
  if (with_lib):
    withLib = Let("map", fileToNano("nano/tests/map.ml"),
      Let("foldl", fileToNano("nano/tests/foldl.ml"),
      e))
  else:
    withLib = e


  ast = withLib.toProlog()

  results = list(p.query("typeof(%s,%s,T)" % (builtins, ast)))
  if (len(results) < 1):
    print("Type error for: %s" % str(e))
    return "Type Error"
  else:
    # print("Typed to %s" % results[0]['T'])
    return results[0]['T']

def strToType(s: str, with_lib=True) -> Type:
  return parseType(exprToType(parseNano(s), with_lib=with_lib))
