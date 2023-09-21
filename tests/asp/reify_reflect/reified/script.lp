program(0,"base",constants(1),statements(2)).
statements(2,0,script(3)).
script(3,python,"\nfrom clingo . symbol import Number\n\ndef divisors (a):\n    a = a.number\n    for i in range (1 , a +1):\n        if a % i == 0:\n        yield Number(i)\n").
