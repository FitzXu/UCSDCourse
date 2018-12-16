#!/usr/bin/env python3
from misc import Failure

class profiled(object):
    def __init__(self,f):
        self.__count=0
        self.__f=f
        self.__name__=f.__name__
    def __call__(self,*args,**dargs):
        self.__count+=1
        return self.__f(*args,**dargs)
    def count(self):
        return self.__count
    def reset(self):
        self.__count=0

class traced(object):
    level = 0
    def __init__(self,f):
        """Initialize the decorator with a given function"""
        self.f = f
        self.__name__ = f.__name__


    def __call__(self, *args, **kargs):
        """
        print an art tree of the recursive calls on the decorated 
        function and their return values.
        """        
        prettify = '| ' * traced.level + ',- ' + self.__name__ + '('
        prettify += ', '.join([repr(arg) for arg in args])
        prettify += ', '.join([k + '=' + repr(v) for k, v in kargs.items()]) + ')'
        print(prettify)
        traced.level += 1
        try:
            result = self.f(*args, **kargs)
            traced.level -= 1
            line = '| ' * traced.level + '`- ' + repr(result)
            print(line)
            return result
        except Exception as e:
            traced.level -= 1
            raise e


class memoized(object):
    def __init__(self,f):
        """Initialize the decorator with a given function"""
        self.f = f
        self.__name__ = f.__name__
        self.mem = {}

    def __call__(self, *args, **kargs):
        """
        Use a dictionary to store the called function name combining with
        the given positional arguments,keyword arguments. If called again, we should return the same
        value or throw the same exception
        """

        # Notice that eg. fun(b=1,c=2) and fun(c=2,b=1) should be the same call
        # So we sort the keywords alphabatically and convert the dictionary into a list such as
        # [("a":1),("b":3)]
        kargs_sort = sorted(zip(list(kargs.keys()),list(kargs.values())),key=lambda x:x[0])
        key = (self.__name__, str(args), str(kargs_sort))

        if key in self.mem:
            result = self.mem[key]
            if isinstance(result, Exception):
                raise result
            else:
                return result
        try:
            result = self.f(*args, **kargs)
            self.mem[key] = result
            return result
        except Exception as e:
            self.mem[key] = e
            raise e            


# run some examples.  The output from this is in decorators.out
def run_examples():
    for f,a in [(fib_t,(7,)),
                (fib_mt,(7,)),
                (fib_tm,(7,)),
                (fib_mp,(7,)),
                (fib_mp.count,()),
                (fib_mp,(7,)),
                (fib_mp.count,()),
                (fib_mp.reset,()),
                (fib_mp,(7,)),
                (fib_mp.count,()),
                (even_t,(6,)),
                (quicksort_t,([5,8,100,45,3,89,22,78,121,2,78],)),
                (quicksort_mt,([5,8,100,45,3,89,22,78,121,2,78],)),
                (quicksort_mt,([5,8,100,45,3,89,22,78,121,2,78],)),
                (change_t,([9,7,5],44)),
                (change_mt,([9,7,5],44)),
                (change_mt,([9,7,5],44)),
                ]:
        print("RUNNING %s(%s):" % (f.__name__,", ".join([repr(x) for x in a])))
        rv=f(*a)
        print("RETURNED %s" % repr(rv))

@traced
def fib_t(x):
    if x<=1:
        return 1
    else:
        return fib_t(x-1)+fib_t(x-2)

@traced
@memoized
def fib_mt(x):
    if x<=1:
        return 1
    else:
        return fib_mt(x-1)+fib_mt(x-2)

@memoized
@traced
def fib_tm(x):
    if x<=1:
        return 1
    else:
        return fib_tm(x-1)+fib_tm(x-2)

@profiled
@memoized
def fib_mp(x):
    if x<=1:
        return 1
    else:
        return fib_mp(x-1)+fib_mp(x-2)

@traced
def even_t(x):
    if x==0:
        return True
    else:
        return odd_t(x-1)

@traced
def odd_t(x):
    if x==0:
        return False
    else:
        return even_t(x-1)

@traced
def quicksort_t(l):
    if len(l)<=1:
        return l
    pivot=l[0]
    left=quicksort_t([x for x in l[1:] if x<pivot])
    right=quicksort_t([x for x in l[1:] if x>=pivot])
    return left+l[0:1]+right

@traced
@memoized
def quicksort_mt(l):
    if len(l)<=1:
        return l
    pivot=l[0]
    left=quicksort_mt([x for x in l[1:] if x<pivot])
    right=quicksort_mt([x for x in l[1:] if x>=pivot])
    return left+l[0:1]+right

class ChangeException(Exception):
    pass

@traced
def change_t(l,a):
    if a==0:
        return []
    elif len(l)==0:
        raise ChangeException()
    elif l[0]>a:
        return change_t(l[1:],a)
    else:
        try:
            return [l[0]]+change_t(l,a-l[0])
        except ChangeException:
            return change_t(l[1:],a)

@traced
@memoized
def change_mt(l,a):
    if a==0:
        return []
    elif len(l)==0:
        raise ChangeException()
    elif l[0]>a:
        return change_mt(l[1:],a)
    else:
        try:
            return [l[0]]+change_mt(l,a-l[0])
        except ChangeException:
            return change_mt(l[1:],a)


