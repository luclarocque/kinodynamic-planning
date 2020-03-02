from subformulas import *


# s is a node of the simplified Kripke structure
def succ(s):
    return s.children

# returns the subformula of the form "muX.(~)" or "nuY.(~)"
def bound_formula(x, PHI):
    sf = subformulas(PHI)
    for f in sf:
        if f[1:4] == "u{}.".format(x):
            return f

####-----------------------------------------------------------------------####

stack = [] # use append/pop to add to end/take from end. Will contain tuples (s, psi).

# Props is a set of atomic propositions. The propositions are one character each, except negated
#   propositions which are prefixed by '!'.
def modelcheck(K, PHI, Props, s):
    spec = nestexp(PHI)  # creates a list from string PHI, separated by spaces and with parens denoting nested lists.
    return mchelper(K, PHI, Props, s, spec)

# e.g., spec = ['nuY.', ['muX.', [['p', '&&', '<>X'], '||', '<>Y']]]
def mchelper(K, PHI, Props, s, spec):
    #global stack
    head, tail = spec[0], spec [1:]
    if isinstance(head, str):
        if head in Props:
            return (head in K.S)  # s is a node of the simplified Kripke structure, with its satisfied props stored in S
        elif (head[0] == "!" and head[1:] in Props):
            return not(head in s.S)
        elif len(spec) == 3 and spec[1] == "&&" and \
          ((head in Props) or (head[0] == "!" and head[1:] in Props)):
            return ( mchelper(K, PHI, Props, s, head) and mchelper(K, PHI, Props, s, tail[1:]) )
        elif len(spec) == 3 and spec[1] == "||":
            return ( mchelper(K, PHI, Props, s, spec[0]) or mchelper(K, PHI, Props, s, spec[2]) )
        elif head[:2] == "<>":
            cur_prop = ""
            if len(head) > 2:
                cur_prop = head[2:]
            else:
                cur_prop = tail[0]
            for nexts in succ(s):
                if mchelper(K, PHI, Props, nexts, cur_prop):
                    return True
            return False
        elif head[:2] == "mu" or head[:2] == "nu":
            stack.append((s, tail[0]))
            value = mchelper(K, PHI, Props, s, tail[0])
            stack.remove((s,tail[0]))
            return value
        elif not(head in Props):   # head is a variable
            bf = bound_formula(head, PHI)
            bf_spec = nestexp(bf[5:-1])  # skips over the "muX.(" and leaves out the final paren
            if (s, bf_spec) in stack:
                if bf[:2] == "mu":
                    return False
                elif bf[:2] == "nu":
                    return True
            else:
                return modelcheck(K, PHI, Props, s, bf_spec)
    # otherwise head is a list
    else:
        if len(spec) == 3 and spec[1] == "||":
            return ( mchelper(K, PHI, Props, s, spec[0]) or mchelper(K, PHI, Props, s, spec[2]) )
        else:
            return mchelper(K, PHI, Props, s, head)
        










