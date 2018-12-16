
class Failure(Exception):
    """Failure exception"""
    def __init__(self,value):
        self.value=value
    def __str__(self):
        return repr(self.value)

class Pair(object):
    """A simple pair of elements"""
    def __init__(self,first=None,second=None):
        """Construct a Pair.  Any arguments not provided are initialized to None"""
        self.first=first
        self.second=second
    def __repr__(self):
        """Return a representation of this Pair object"""
        return "%s(%s, %s)" % (self.__class__.__name__,repr(self.first),repr(self.second))


def compose2(f,g):
    """Compose two function"""
    def composition(*args,**dargs):
        return f(g(*args,**dargs))
    return composition

def compose(*fns):
    """Compose an arbitrary number of functions"""
    return reduce(lambda a,b:compose2(b,a),reversed(list(fns)+[lambda x:x]))



