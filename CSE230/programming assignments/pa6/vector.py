from misc import Failure

class Vector(object):
    def __init__(self, args):
      """
      constructor
      Input: int means the length of the vector or a list of values to assign to the vetor
      Out: vector
      """
      if (isinstance(args,int)):
        if (args < 0):
          raise ValueError("Vector length must greater than 0.")
        self.vec = [0.0] * args
      elif (isinstance(args, list)):
        self.vec = list(args)
      else:
        raise TypeError("The type is expecting is an Integer or a list but given with {}".format(type(args)))
    
    def __repr__(self):
        """ repr is the string represention of the class, it returns
        Vector(contents of the list)"""
        return "Vector(" + repr(self.vec) + ")"

    def __len__(self):
      """
      compute the length of vector
      """
      return len(self.vec)
    
    def __iter__(self):
      """ Returns an iterator for the vector """
      for item in self.vec:
        yield(item)
        
    def __add__(self,second):
      """
      This function override the function object.__add__() to implement binary operation of addition.
      """
      return Vector(([x + y for x, y in zip(list(self), list(second))]))
    
    
    def __iadd__(self, second):
      self.vec = Vector(([x + y for x, y in zip(list(self), list(second))]))
      return self.vec
    
    def __radd__(self,second):
      return Vector(([x + y for x, y in zip(list(self), list(second))]))

    def dot(self, second):
       """
       This function is the implemention of dot product of vector 
       """
       return sum([ x * y for x, y in zip(self, second)])
    def __getitem__ (self, index):
	    """
	    this function get corresponding elems                                          
	    """
	    return self.vec[index]
    def __setitem__ (self, x, y): 
      temp = len(self.vec)    
      self.vec[x] = y
      if ( temp != len(self.vec)):
        raise ValueError(" Can not change length of vector")
    
    def __eq__(self, second):
      if not isinstance(second, Vector):
        return False
      allequal =  ([ x == y for x, y in zip(self, second)])
      if False in allequal:
        return False
      else:
        return True

    def __ne__(self, other):
      return not self.__eq__(other)
      
    def __ge__(self, second):
      if self.__gt__(second):
        return True
      selfSorted = sorted(self, reverse = True)
      secondSorted= sorted(second, reverse = True)
      if selfSorted.__eq__(secondSorted):
        return True
      return False

    def __gt__(self, second):
      if not isinstance(second,Vector):
        return (self > other)
      selfSorted = sorted(self, reverse = True)
      secondSorted= sorted(second, reverse = True)
      for x, y in zip(selfSorted, secondSorted):
        if x > y:
          return True
        elif x < y:
          return False
      return False
      
    def __lt__(self, second):
      return not self.__ge__(second)
    
    def __le__(self, second):
        return not self.__gt__(second)



