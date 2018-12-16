#!/usr/bin/env python3
from decorators import *
from nanoTypes import *
from subprocess import run

try:
  # nano examples
  expected = {
    't1'  : IntTy(),
    't2'  : IntTy(),
    't3'  : IntTy(),
    't5'  : IntTy(),
    't6'  : IntTy(),
    't7'  : IntTy(),
    't8'  : IntTy(),
    't9'  : IntTy(),
    't10'  : IntTy(),
    't11'  : IntTy(),
    't12'  : IntTy(),
    't13'  : IntTy(),
    't14'  : ListTy(IntTy()),
    't15'  : IntTy(),
    't16'  : ListTy(IntTy()),
    't17'  : IntTy(),
    't18'  : IntTy(),
    't19'  : ListTy(IntTy()),
    't20'  : IntTy()      
  }
  
  for suf, t in expected.items():
    fname = "nano/tests/%s.ml" % suf
    print("Typechecking %s to %s" % (fname, t))
    fcontents = run(["cat", fname], stdout=PIPE, universal_newlines=True, check=True)
    ty = strToType(fcontents.stdout)

    if (ty != t):
      print("error: typed to %s instead of %s" % (str(ty), str(t)))
  # decorators
  run_examples()

except Exception as e:
  print("Exception in examples: %s" % str(e))
  raise e
