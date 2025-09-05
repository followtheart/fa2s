# FA2S LR Parser Generator

This project turns a context-free grammar (in a simple BNF) into a standalone FA2S program (.fa2s) which performs SLR(1) LR parsing on an input token stream (single-character tokens, ending with `$`).

Design
- Runtime target: FA2S (finite automaton + 2 stacks). The generated FA2S code:
  - Stack A: LR state stack (each state encoded as one character from 0-9A-Za-z).
  - Stack B: Grammar symbol stack (terminals and nonterminals as their literal characters).
  - SHIFT: read a token, push it to B, push target LR state to A, then jump to state S<target>.
  - REDUCE: pop |RHS| items from both stacks, push LHS to B, then do GOTO by inspecting A-top and push target state, jump to S<goto>.
  - ACCEPT: on `$` in ACCEPT configuration -> `halt:accept`.

Grammar format
- Lines: `NonTerm -> alternative1 | alternative2 | ...`
- Tokens in RHS are space-separated.
- Nonterminals: uppercase identifiers (E, T, F, ...).
- Terminals: single characters (i, +, *, (, ), ...). Use `$` as end-of-input.
- Epsilon production: use the single token `ε` (Greek letter epsilon).

Example grammar (SLR(1) for classic expression):
```
E -> E + T | T
T -> T * F | F
F -> ( E ) | i
```

Usage
- Generate FA2S parser:
  - `python3 tools/slr_to_fa2s.py examples/expr.g > out/expr.fa2s`
- Run with FA2S interpreter (must end input with `$`):
  - `./fa2s out/expr.fa2s --input "i+i*i$"`
  - Exit code: 0 accept, 1 reject.

Notes
- The generator builds LR(0) item sets and SLR(1) parse tables. It fails if there are shift/reduce or reduce/reduce conflicts.
- State encoding supports up to 62 LR states (0-9A-Za-z).
- Generated FA2S aims for clarity over compactness; it emits per-state SHIFT/ACCEPT rules and shared REDUCE/GOTO subroutines.

Files
- `tools/slr_to_fa2s.py`: grammar -> FA2S generator (SLR(1))
- `examples/expr.g`: sample grammar
- Output: `out/*.fa2s` generated parser programs