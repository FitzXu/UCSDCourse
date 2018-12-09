#!/usr/bin/env python3
"""--------CATEGORY 1: list comprehensions and functional python--------"""

# Spring 13, 5) reverse
def rev(l):
  return [l[index] for index in range(len(l)-1,-1,-1)]

# Winter 13, 3) matrices
def transpose(m):
  height = len(m)
  width = len(m[0])
  return [ [  m[i][j] for i in range(height)] for j in range(width)]

# FA 13, 3) dictionaries
def lookup(d,k):
  return [item[1] for item in d if item[0]==k]

def cond(b, t, f):
  if b: return t
  else: return f

def update(d,k,v):
  return [(item[0],cond(item[0]==k,v,item[1])) for item in d]

def delete(d,k):
  return [item for item in d if item[0]!=k]

def add(d,k,v):
  return d.append((k,v))

def update(d,k,v):
  newL = []
  for item in d:
    if k == item[0]:
      newL.append((k,v))
    else:
      newL.append(item)

"""--------CATEGORY 2: decorators--------"""
# FA 13, 4) in-range
# Spring 13, 6) print-some
# Winter 13, 5) lift to array
# Winter 12, 3) derivative

# class based

# @derivative(delta)
 # def double(x): return 2 * x
def derivative(delta):
  class derivative_inner(object):
    def __init__(self,f):
      self.__f = f
      self.__delta = delta
    def __call__(self, *args, **kargs):
      numer = self.__f(
          *[a + self.__delta for a in args],
          **{k:(v+self.__delta) for k,v in kargs.items()})
      numer = numer - self.__f(*args,**kargs)
      
      return round(numer / self.__delta,2)
  return derivative_inner

@derivative(0.0001)
def double_class(x): return 2 * x

# functional
def derivative_func(delta):
  def derivative_inner(f):
    def deriv_most_inner(*args, **kargs):
      numer = f(
          *[a + delta for a in args],
          **{k:(v+delta) for k,v in kargs.items()})
      numer = numer - f(*args,**kargs)
        
      return round(numer / delta,2)
    return deriv_most_inner
  return derivative_inner

@derivative_func(0.0001)
def double_func(x): return 2 * x

# Winter 11, 3) print_args
"""--------CATEGORY 3: complicated datastructure--------"""
# Spring 13, 7) prolog unification

# Winter 13, 4) Game of Life

# Winter 12, 2) images

# Winter 11, 4) images
