import re


class Failure(Exception):
    """Failure exception"""
    def __init__(self,value):
        self.value=value
    def __str__(self):
        return repr(self.value)

# Problem 1
def closest_to(l,v):
    """
    Returns the element of the list l closest in value to v.
    In the case of a tie, the first such element is returned. If l is empty, None is returned. 
    """
    if l is None:
        return None
    diff = abs(l[0] - v)
    res = None
    for item in l[1:]:
        if(abs(item-v)<diff):
            diff = abs(item-v)
            res = item
    return res

def make_dict(keys,values):
    """
    Return a dictionary pairing corresponding keys to values.
    """
    d = {}
    for index,key in enumerate(keys):
        d[key] = values[index]
    return d
   
# file IO functions
def word_count(fn):
    """
    Open the file fn and return a dictionary mapping words to the number
    of times they occur in the file.  A word is defined as a sequence of
    alphanumeric characters and underscore.  All spaces and punctuation are ignored.
    Words are returned in lower case
    """
    with open(fn) as f:
        content = f.read()
    d = {}
    pattern = re.compile('[A-Za-z0-9_]+')
    words = "".join([char if pattern.match(char) else ' ' for char in content.lower()])
    for w in words.split():
        if w in d:
            d[w] += 1
        else:
            d[w] = 1
    return d


