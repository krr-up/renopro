#script (python)
from clingo . symbol import Number

def divisors (a):
    a = a.number
    for i in range (1 , a +1):
        if a % i == 0:
        yield Number(i)
#end.
