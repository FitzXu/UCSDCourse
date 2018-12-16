#!/usr/bin/env python3

from misc import Failure

# prologable interface
class Prologable():
    def toProlog(self) -> str:        
        raise Failure("SHOULD NOT GET HERE -- subclasses should override")

    def __eq__(self,  other):
        if isinstance(other,  Prologable):
            return self.toProlog() == other.toProlog()
        else:
            return False

    def __str__(self):
        return self.toProlog()


# expression interface
class Expression(Prologable):
    pass
# binop interface

class Bop(Prologable):
    pass

class Plus(Bop):    
    def toProlog(self) -> str:        
        """toProlog: Plus Binary Operation"""
        return "plus"

class Minus(Bop):    
    def toProlog(self) -> str:        
        """toProlog: Minus Binary Operation"""
        return "minus"

class Mul(Bop):
    def toProlog(self) -> str:        
        """toProlog: Multiply Binary Operation"""
        return "mul"

class Div(Bop):
    def toProlog(self) -> str:
        """toProlog: Divide Binary Operation"""
        return "div"

class Eq(Bop):
    def toProlog(self) -> str:        
        """toProlog: Comparison Binary Operation for equal operator"""
        return "eq"

class Neq(Bop):    
    def toProlog(self) -> str:        
        """toProlog: Comparison Binary Operation for non-equal operator"""
        return "neq"

class Lt(Bop):
    def toProlog(self) -> str:
        """toProlog: Comparison Binary Operation for non-equal operator"""
        return "lt"

class Leq(Bop):
    def toProlog(self) -> str:
        """toProlog: Comparison Binary Operation for less and equal operator"""
        return "leq"

class And(Bop):
    def toProlog(self) -> str:
        """toProlog: Boolean Binary Operation for and operator"""
        return "and"

class Or(Bop):
    def toProlog(self) -> str:        
        """toProlog: Boolean Binary Operation for or operator"""
        return "or"

class Cons(Bop):
    def toProlog(self) -> str:
        """toProlog: Boolean Binary Operation for cons"""
        return "cons"

# Expressions
class Const(Expression):
    def __init__(self,  i: int):
        self.v = i
    def toProlog(self) -> str:
        """toProlog: Boolean Binary Operation for cons operator of list"""
        return "const("+str(self.v)+")"

class Bool(Expression):
    def __init__(self,  b: bool):
        self.v = b
    def toProlog(self) -> str:
        """toProlog: Bool type expression"""
        return "boolean("+str(self.v)+")"

class NilExpr(Expression):
    def __init__(self):
        return
    def toProlog(self) -> str: 
        """toProlog: Nil type expression"""
        return "nil"

class Var(Expression):
    def __init__(self,  v: str):
        self.v = v
    def toProlog(self) -> str: 
        """toProlog: Variable type expression"""
        return "var("+str(self.v)+")"
    
class Bin(Expression):
    def __init__(self,  l: Expression,  o: Bop,  r:Expression):
        self.l = l
        self.r = r
        self.o = o
    def toProlog(self) -> str:
        """toProlog: Variable type expression"""
        return "bin("+(self.l).toProlog()+", "+(self.o).toProlog()+", "+(self.r).toProlog()+")"
    
class If(Expression):
    def __init__(self,  c: Expression,  t: Expression,  f: Expression):
        self.c = c
        self.t = t
        self.f = f
    def toProlog(self) -> str: 
        """toProlog: If expression type
        c is the condition expression
        t is the the statement
        f is the the statement"""        
        return "ite("+(self.c).toProlog()+", "+(self.t).toProlog()+", "+(self.f).toProlog()+")"

class Let(Expression):
    def __init__(self,  v: str,  e: Expression,  body: Expression):
        self.v = v
        self.e = e
        self.body = body
    def toProlog(self) -> str:
        """toProlog: Let statement
        v is varaiable name
        e is expression
        f is the the statement"""
        return "let("+self.v+", "+(self.e).toProlog()+", "+(self.body).toProlog()+")"

class Letrec(Expression):
    def __init__(self,  v: str,  e: Expression,  body: Expression):
        self.v = v
        self.e = e
        self.body = body
    def toProlog(self) -> str:
        """toProlog: Letrec statement
        v is varaiable name
        e is expression
        body is the the statement"""
        return "letrec("+self.v+", "+(self.e).toProlog()+", "+(self.body).toProlog()+")"

class App(Expression):
    def __init__(self,  f: Expression,  arg: Expression):
        self.f = f
        self.arg = arg
    def toProlog(self) -> str:
        """toProlog: app statement
        f is function that we need to be applied
        arg is arguments for the function"""
        return "app("+self.f.toProlog()+", "+self.arg.toProlog()+")"
class Fun(Expression):
    def __init__(self,  v: str,  body: Expression):
        self.v = v
        self.body = body
    def toProlog(self) -> str:
        """toProlog: function statement
        v is parameter
        body is function body"""
        return "fun("+self.v+", "+self.body.toProlog()+")"


# Types
class Type(Prologable):
    pass
class IntTy(Type):
    def __init__(self):
        return
    def toProlog(self) -> str:
        """toProlog: IntTy statement
        v is parameter
        body is function body"""
        return "int"

class BoolTy(Type):
    def __init__(self):
        return
    def toProlog(self) -> str:
        """toProlog: Boolty statement
        v is parameter
        body is function body
        """ 
        return "bool"

class ArrowTy(Type):
    def __init__(self,  l: Type,  r: Type):
        self.l = l
        self.r = r
    def toProlog(self) -> str: 
        """toProlog: ArrowTy statement
        l is left type, which is the input type
        r is right type, which is the input type
        """ 
        return "arrow("+self.l.toProlog()+", "+self.r.toProlog()+")"

class ListTy(Type):
    def __init__(self,  inner: Type):
        self.inner = inner
    def toProlog(self) -> str:
        """toProlog: ArrowTy statement
        l is left type, which is the input type
        r is right type, which is the input type
        """         
        return "list("+self.inner.toProlog()+")"

class VarTy(Type):
    def __init__(self,  name: str):
        self.name = name
    def toProlog(self) -> str: 
        return self.name

