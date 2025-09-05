# Classic expression grammar (SLR(1))
# Nonterminals: E T F
# Terminals: i + * ( ) and end-of-input $
E -> E + T | T
T -> T * F | F
F -> ( E ) | i