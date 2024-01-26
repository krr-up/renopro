#script (python)

from clingo.symbol import String, Function

def string_to_const(string: String):
    return Function(string.string, [])

#end.
