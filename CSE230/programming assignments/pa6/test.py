import misc
import vector
import sys

KEY="130>>"

def run_test(fun,args,chk,pts):
    try:
        rv=fun(*args)
        print("%s calling %s(%s)... " % \
              (KEY,fun.__name__,",".join([repr(x) for x in args])),rv)
    except Exception as e:
        print("Exception: "+str(e))
        return 0
    else:
        if chk(rv):
            print("Good")
            return pts
        else:
            print("Wrong")
            return 0

def chk_val(x):
    return lambda y: x==y

def chk_flt(x):
    return lambda y: abs(x-y)<1e-8

def chk_dict(qry,rslt,sz):
    return lambda d: len(d)==sz and qry in d and d[qry]==rslt

def chk_set(qry,sz):
    return lambda d: len(d)==sz and qry in d
    
def chk_repr(str):
    return lambda d: repr(d)==str

def chk_file(fn):
    return lambda d: sys.stdout.write("\nSee file \"%s\"\n" % fn) or True

def run_tests(tests):
    score=0
    total=0
    for (fun,args,chk,pts) in tests:
        score+=run_test(fun,args,chk,pts)
        total+=pts
    return (score,total)

def run_all_tests():
    globals={}
    def test_1a_a():
        try:
            vector.Vector(-4)
            return False
        except ValueError:
            return True
    def test_1c_a():
        v=vector.Vector(["f","b"])
        v+=("oo","oo")
        return v
    def test_1e_a():
        v=vector.Vector(7)
        return v[4]
    def test_1e_b():
        v=vector.Vector(7)
        v[4]="foo"
        return v[4]
    def test_1e_c():
        v=vector.Vector(7)
        v[4]="foo"
        return v
    return run_tests([
        #problem 1
        (misc.closest_to,[[2,4,8,9],7],chk_val(8),1),
        (misc.closest_to,[[2,4,8,9],5],chk_val(4),1),
        (misc.make_dict,[["foo","baz"],["bar","blah"]],chk_val({'foo': 'bar', 'baz': 'blah'}),1),
        (misc.make_dict,[[1],[100]],chk_val({1: 100}),1),
        (misc.word_count,["news.txt"],chk_dict("edu",2,407),1),

        #problem 2a
        (vector.Vector,[3],chk_repr("Vector([0.0, 0.0, 0.0])"),1),
        (vector.Vector,[[4.5,"foo",0]],
         chk_repr("Vector([4.5, 'foo', 0])"),1),
        (vector.Vector,[0],chk_repr("Vector([])"),1),
        (test_1a_a,[],chk_val(True),1),

        #problem 2b
        (lambda: [x*2 for x in vector.Vector([3,3.25,"foo"])],[],
         chk_val([6,6.5,"foofoo"]),1),
        (lambda: len(vector.Vector(23)),[],chk_val(23),1),

        #problem 2c
        (lambda: vector.Vector([6,8,2])+vector.Vector([4,-3,2]),[],
         chk_repr("Vector([10, 5, 4])"),1),
        (lambda: vector.Vector([6,8,2])+[4,-3,2],[],
         chk_repr("Vector([10, 5, 4])"),1),
        (lambda: (6,8,2)+vector.Vector([4,-3,2]),[],
         chk_repr("Vector([10, 5, 4])"),1),
        (test_1c_a,[],chk_repr("Vector(['foo', 'boo'])"),1),

        #problem 2d
        (lambda: vector.Vector([6,8,2]).dot(vector.Vector([4,-3,2])),[],
         chk_val(4),1),
        (lambda: vector.Vector([6,8,2]).dot([4,-3,2]),[],
         chk_val(4),1),

        #problem 2e
        (test_1e_a,[],chk_val(0.0),1),
        (test_1e_b,[],chk_val("foo"),1),
        (test_1e_c,[],
         chk_repr("Vector([0.0, 0.0, 0.0, 0.0, 'foo', 0.0, 0.0])"),1),

        #problem 2f
        (lambda: (lambda a,b,c:a<b)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(False),1),
        (lambda: (lambda a,b,c:a>b)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(False),1),
        (lambda: (lambda a,b,c:a>=b)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(True),1),
        (lambda: (lambda a,b,c:a>c)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(False),1),
        (lambda: (lambda a,b,c:a<c)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(True),1),
        (lambda: (lambda a,b,c:a>=c)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(False),1),
        (lambda: (lambda a,b,c:a==c)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(False),1),
        (lambda: (lambda a,b,c:a==a)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(True),1),
        (lambda: (lambda a,b,c:a!=c)(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(True),1),
        (lambda: (lambda a,b,c:a!=[1,3,5])(vector.Vector([1,3,5]),
                                    vector.Vector([5,1,3]),
                                    vector.Vector([4,5,4])),[],
         chk_val(True),1),
        
        ])

(s,t)=run_all_tests()
print("%s Results: (%d/%d)" % (KEY,s,t))
print("%s Compiled" % KEY)
    
