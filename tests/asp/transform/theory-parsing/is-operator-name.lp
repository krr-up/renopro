#script (python)

from clingo.symbol import String, Number

def is_operator_name(name: String):
    name_str = name.string																	
    return Number(1) if (name_str and name_str[0] in "/!<=>+-*\\?&@|:;~^.") or (name_str == "not") else Number(0)

#end.
