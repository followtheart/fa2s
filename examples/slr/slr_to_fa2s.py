#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SLR(1) LR parser generator targeting FA2S
# Input: grammar in simple BNF
# Output: a standalone .fa2s program that parses single-char token streams (ending with $)

import sys
import re
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional

try:
    from dataclasses import dataclass
except ImportError:
    print("Error: Need Python 3.7+ for dataclasses", file=sys.stderr)
    sys.exit(1)

SYMS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"  # 62 symbols

def die(msg: str):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(2)

def is_nonterminal(tok: str) -> bool:
    # Uppercase identifier (letters/_digits allowed), starts with uppercase letter
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", tok))

def is_terminal(tok: str) -> bool:
    # single character terminal (including punctuation), excluding ε
    return len(tok) == 1 and tok != "ε"

def escape_char_for_fa2s(ch: str) -> str:
    # single-quoted char literal content
    if ch == "\\": return "\\\\"
    if ch == "'": return "\\'"
    if ch == "\n": return "\\n"
    if ch == "\t": return "\\t"
    if ch == "\r": return "\\r"
    return ch

class Grammar:
    def __init__(self):
        self.start_sym: Optional[str] = None
        self.prods: List[Tuple[str, List[str]]] = []  # (LHS, RHS tokens)
        self.nonterms: Set[str] = set()
        self.terms: Set[str] = set()
        self.has_epsilon: bool = False

    @staticmethod
    def parse(text: str) -> "Grammar":
        g = Grammar()
        for ln, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                die(f"Line {ln}: missing '->'")
            lhs, rhs = line.split("->", 1)
            lhs = lhs.strip()
            if not is_nonterminal(lhs):
                die(f"Line {ln}: LHS must be uppercase nonterminal, got '{lhs}'")
            if g.start_sym is None:
                g.start_sym = lhs
            alts = [alt.strip() for alt in rhs.split("|")]
            for alt in alts:
                if alt == "" or alt == "ε":
                    rhs_toks = ["ε"]
                    g.has_epsilon = True
                else:
                    rhs_toks = alt.split()
                for t in rhs_toks:
                    if t != "ε" and not (is_nonterminal(t) or is_terminal(t)):
                        die(f"Line {ln}: bad token '{t}' (use uppercase for nonterminals; single char for terminals)")
                g.prods.append((lhs, rhs_toks))
        if g.start_sym is None:
            die("Empty grammar")
        # collect symbol sets
        for lhs, rhs in g.prods:
            g.nonterms.add(lhs)
            for t in rhs:
                if t == "ε": continue
                if is_nonterminal(t): g.nonterms.add(t)
                else: g.terms.add(t)
        g.terms.add("$")  # end marker
        return g

# FIRST and FOLLOW for SLR(1)
def compute_first(g: Grammar) -> Dict[str, Set[str]]:
    FIRST: Dict[str, Set[str]] = {X: set() for X in g.nonterms}
    for a in g.terms:
        FIRST[a] = {a}
    changed = True
    while changed:
        changed = False
        for A, rhs in g.prods:
            if rhs == ["ε"]:
                if "ε" not in FIRST[A]:
                    FIRST[A].add("ε"); changed = True
                continue
            # FIRST of sequence
            nullable_prefix = True
            for X in rhs:
                for sym in FIRST.get(X, set()):
                    if sym != "ε" and sym not in FIRST[A]:
                        FIRST[A].add(sym); changed = True
                if "ε" not in FIRST.get(X, set()):
                    nullable_prefix = False
                    break
            if nullable_prefix:
                if "ε" not in FIRST[A]:
                    FIRST[A].add("ε"); changed = True
    return FIRST

def compute_follow(g: Grammar, FIRST: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    FOLLOW: Dict[str, Set[str]] = {X: set() for X in g.nonterms}
    FOLLOW[g.start_sym].add("$")
    changed = True
    while changed:
        changed = False
        for A, rhs in g.prods:
            trailer = set(FOLLOW[A])
            for X in reversed(rhs):
                if is_nonterminal(X):
                    for sym in trailer:
                        if sym not in FOLLOW[X]:
                            FOLLOW[X].add(sym); changed = True
                    if "ε" in FIRST.get(X, set()):
                        trailer = trailer.union(FIRST[X] - {"ε"})
                    else:
                        trailer = FIRST[X] - {"ε"}
                else:
                    trailer = FIRST[X]
    return FOLLOW

# LR(0) items
@dataclass(frozen=True)
class Item:
    lhs: str
    rhs: Tuple[str, ...]
    dot: int


def closure(items: Set[Item], g: Grammar) -> Set[Item]:
    C = set(items)
    changed = True
    while changed:
        changed = False
        new_items = set()
        for it in C:
            if it.dot < len(it.rhs):
                X = it.rhs[it.dot]
                if is_nonterminal(X):
                    for (A, rhs) in g.prods:
                        if A == X:
                            cand = Item(A, tuple(rhs if rhs != ["ε"] else ()), 0)
                            if cand not in C:
                                new_items.add(cand)
        if new_items:
            C |= new_items
            changed = True
    return C

def goto(items: Set[Item], X: str, g: Grammar) -> Set[Item]:
    moved = set()
    for it in items:
        if it.dot < len(it.rhs) and it.rhs[it.dot] == X:
            moved.add(Item(it.lhs, it.rhs, it.dot + 1))
    if not moved: return set()
    return closure(moved, g)

def canon_lr0_sets(g: Grammar) -> Tuple[List[Set[Item]], Dict[Tuple[int, str], int]]:
    # augmented grammar S' -> S
    S_dash = g.start_sym + "'"
    while S_dash in g.nonterms:
        S_dash += "'"
    aug_prod = (S_dash, [g.start_sym])
    prods_aug = [aug_prod] + g.prods
    # temporarily build an augmented grammar for closure/goto
    g_aug = Grammar()
    g_aug.start_sym = S_dash
    g_aug.prods = prods_aug
    g_aug.nonterms = set(g.nonterms) | {S_dash}
    g_aug.terms = set(g.terms)

    I0 = closure({Item(S_dash, tuple([g.start_sym]), 0)}, g_aug)
    C = [I0]
    trans: Dict[Tuple[int, str], int] = {}
    work = deque([0])
    while work:
        i = work.popleft()
        syms = set()
        for it in C[i]:
            if it.dot < len(it.rhs):
                syms.add(it.rhs[it.dot])
        for X in syms:
            Ij = goto(C[i], X, g_aug)
            if not Ij: continue
            # find if already exists
            found = None
            for idx, st in enumerate(C):
                if st == Ij:
                    found = idx; break
            if found is None:
                found = len(C)
                C.append(Ij)
                work.append(found)
            trans[(i, X)] = found
    return C, trans, S_dash, prods_aug

def encode_state(idx: int) -> str:
    if idx >= len(SYMS):
        die(f"Too many LR states ({idx}); max supported is {len(SYMS)}")
    return SYMS[idx]

def build_slr_tables(g: Grammar):
    FIRST = compute_first(g)
    FOLLOW = compute_follow(g, FIRST)
    C, trans, S_dash, prods_aug = canon_lr0_sets(g)

    # Map production to index (skip augmented at index 0)
    prod_list = prods_aug  # index 0 is S'->S
    # ACTION and GOTO
    ACTION: Dict[Tuple[int, str], Tuple[str, int]] = {}  # (state, term) -> ('S',j) | ('R',k) | ('A',-)
    GOTO: Dict[Tuple[int, str], int] = {}               # (state, NonT) -> j

    # fill goto for NonT
    for (i, X), j in trans.items():
        if is_nonterminal(X):
            GOTO[(i, X)] = j

    # shifts
    for (i, X), j in trans.items():
        if X in g.terms:
            if (i, X) in ACTION:
                die(f"Shift conflict at state {i}, term {X}")
            ACTION[(i, X)] = ('S', j)

    # reductions and accept
    for i, I in enumerate(C):
        for it in I:
            if it.dot == len(it.rhs):
                if it.lhs == S_dash:
                    # accept on $
                    if (i, "$") in ACTION and ACTION[(i, "$")] != ('A', -1):
                        die(f"Conflict at state {i}, on $")
                    ACTION[(i, "$")] = ('A', -1)
                else:
                    # reduce A->rhs on FOLLOW(A)
                    k = None
                    # find production index k in original (1-based non-augmented)
                    for idx, (A, rhs) in enumerate(prod_list):
                        if A == it.lhs and tuple(rhs if rhs != ["ε"] else ()) == it.rhs:
                            k = idx
                            break
                    if k is None:
                        die("Internal error: production not found for reduce")
                    # NB: k>=1 since idx 0 is S'->S
                    for a in FOLLOW[it.lhs]:
                        if (i, a) in ACTION and ACTION[(i, a)] != ('R', k):
                            die(f"SLR conflict at state {i}, on lookahead {a}")
                        ACTION[(i, a)] = ('R', k)

    return ACTION, GOTO, C, prod_list, S_dash

def emit_fa2s(g: Grammar, ACTION, GOTO, C, prod_list, S_dash):
    out = []
    def w(s=""):
        out.append(s)

    # Helpers
    def lit_guard(ch: str) -> str:
        return f"lit:'{escape_char_for_fa2s(ch)}'"

    state_chars = [encode_state(i) for i in range(len(C))]
    state_name = lambda i: f"S{state_chars[i]}"

    # Header
    w(".start init")
    w("")
    # init: push initial state '0' (encoded char of state 0)
    w(f"init : eps, A=empty, B=empty -> pushA:'{escape_char_for_fa2s(state_chars[0])}' ; {state_name(0)}")
    w("")

    # Shared REDUCE subroutines by production index (>=1)
    # prod_list[idx] = (LHS, RHS_list), idx=0 is S'->S
    used_lhs = set()
    for k in range(1, len(prod_list)):
        A, rhs = prod_list[k]
        used_lhs.add(A)
        L = len([] if rhs == ['ε'] else rhs)
        sub = f"RED{k}"
        # entry point: pop L pairs, then push LHS, then goto on LHS
        if L == 0:
            w(f"{sub} : eps, A=*, B=* -> pushB:'{escape_char_for_fa2s(A)}' ; GOTO_{A}")
        else:
            # Main entry point that starts the reduction chain
            w(f"{sub} : eps, A=*, B=* -> ; {sub}_POP{L}")
            # Create a chain of pops
            for i in range(L, 0, -1):
                cur = f"{sub}_POP{i}"
                nxt = f"{sub}_POP{i-1}" if i > 1 else f"{sub}_PUSH"
                w(f"{cur} : eps, A=*, B=* -> popB, popA ; {nxt}")
            w(f"{sub}_PUSH : eps, A=*, B=* -> pushB:'{escape_char_for_fa2s(A)}' ; GOTO_{A}")
        w("")

    # GOTO subroutines by LHS nonterminal: check A-top and push next state
    for A in sorted(used_lhs):
        w(f"GOTO_{A} : eps, A=*, B=* -> ; GOTO_{A}_DISPATCH")
        # we need transitions for each possible A-top state where GOTO is defined
        # Build reverse index: for any i with GOTO(i, A)=j
        pairs = [(i, j) for (i, X), j in GOTO.items() if X == A]
        if not pairs:
            die(f"No GOTO entries for nonterminal {A}")
        for (i, j) in pairs:
            si = state_chars[i]; sj = state_chars[j]
            w(f"GOTO_{A}_DISPATCH : eps, A=lit:'{escape_char_for_fa2s(si)}', B=* -> pushA:'{escape_char_for_fa2s(sj)}' ; {state_name(j)}")
        # safety net: if no match, reject
        w(f"GOTO_{A}_DISPATCH : eps, A=*, B=* -> halt:reject")
        w("")

    # Per-state driver rules (SHIFT/REDUCE/ACCEPT/Error)
    # IMPORTANT: Order: all SHIFTs first, then REDUCEs, then ACCEPT, finally fallback reject
    for i in range(len(C)):
        si = state_chars[i]
        w(f"# ---- LR driver for state {i} (A-top '{si}') ----")
        # SHIFTs
        for (s, a), act in ACTION.items():
            if s != i: continue
            if act[0] == 'S':
                j = act[1]
                sj = state_chars[j]
                w(f"{state_name(i)} : {lit_guard(a)}, A=lit:'{escape_char_for_fa2s(si)}', B=* -> read, pushB:last, pushA:'{escape_char_for_fa2s(sj)}' ; {state_name(j)}")
        # REDUCEs
        for (s, a), act in ACTION.items():
            if s != i: continue
            if act[0] == 'R':
                k = act[1]
                # Do not consume input; jump into REDk subroutine
                w(f"{state_name(i)} : {lit_guard(a)}, A=lit:'{escape_char_for_fa2s(si)}', B=* -> ; RED{k}")
        # ACCEPT
        if (i, "$") in ACTION and ACTION[(i, "$")][0] == 'A':
            w(f"{state_name(i)} : {lit_guard('$')}, A=lit:'{escape_char_for_fa2s(si)}', B=* -> halt:accept")
        # Fallback error
        w(f"{state_name(i)} : any, A=lit:'{escape_char_for_fa2s(si)}', B=* -> halt:reject")
        w("")

    print("\n".join(out))

def main():
    if len(sys.argv) != 2:
        print("Usage: slr_to_fa2s.py <grammar-file>", file=sys.stderr)
        sys.exit(2)
    text = open(sys.argv[1], "r", encoding="utf-8").read()
    g = Grammar.parse(text)
    ACTION, GOTO, C, prod_list, S_dash = build_slr_tables(g)
    emit_fa2s(g, ACTION, GOTO, C, prod_list, S_dash)

if __name__ == "__main__":
    main()