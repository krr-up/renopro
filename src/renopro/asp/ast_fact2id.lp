#script (python)

from clingo.symbol import Symbol, Function

def ast_fact2id(fact: Symbol) -> Symbol:
    return Function(fact.name, fact.arguments[:1])

def none(symb):
		return None

#end.    
