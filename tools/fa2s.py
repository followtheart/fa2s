#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import re

class ParseError(Exception):
    pass

class RuntimeFault(Exception):
    pass

def unescape_char(s: str) -> str:
    # s: content inside single quotes, length >=1 after parsing
    escapes = {
        'n': '\n', 't': '\t', '\\': '\\', "'": "'", '"': '"', 'r': '\r', '0': '\0'
    }
    if len(s) == 0:
        raise ParseError("empty char literal")
    if s[0] == '\\':
        if len(s) == 1:
            raise ParseError("bad escape in char literal")
        c = s[1]
        return escapes.get(c, c)
    if len(s) != 1:
        # take first char, ignore rest
        return s[0]
    return s

def unescape_string(s: str) -> str:
    # s includes quotes "..."
    if len(s) < 2 or s[0] != '"' or s[-1] != '"':
        raise ParseError(f'bad string literal: {s}')
    s = s[1:-1]
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\':
            i += 1
            if i >= len(s):
                raise ParseError("bad string escape")
            e = s[i]
            if e == 'n': out.append('\n')
            elif e == 't': out.append('\t')
            elif e == 'r': out.append('\r')
            elif e == '"': out.append('"')
            elif e == '\\': out.append('\\')
            elif e == '0': out.append('\0')
            else:
                # unknown escape -> literal
                out.append(e)
        else:
            out.append(c)
        i += 1
    return ''.join(out)

def split_actions(s: str):
    # split by comma, but ignore commas inside quotes
    parts = []
    buf = []
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"':
            buf.append(c)
            i += 1
            # toggle string until closing quote (with escapes)
            while i < len(s):
                buf.append(s[i])
                if s[i] == '"' and s[i-1] != '\\':
                    i += 1
                    break
                i += 1
            continue
        if c == ',':
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append(''.join(buf).strip())
    # filter empty
    return [p for p in parts if p]

def parse_char_token(tok: str) -> str:
    # formats: lit:'x'
    m = re.fullmatch(r"lit:'(.*)'", tok)
    if not m:
        raise ParseError(f"bad char token: {tok}")
    return unescape_char(m.group(1))

def parse_guard_in(tok: str):
    # eps | eof | any | lit:'x'
    tok = tok.strip()
    if tok in ('eps', 'eof', 'any'):
        return ('kind', tok, None)
    if tok.startswith("lit:'"):
        ch = parse_char_token(tok)
        return ('kind', 'lit', ch)
    raise ParseError(f"bad IN guard: {tok}")

def parse_guard_top(tok: str):
    # * | empty | lit:'x'
    tok = tok.strip()
    if tok in ('*', 'empty'):
        return ('kind', tok, None)
    if tok.startswith("lit:'"):
        ch = parse_char_token(tok)
        return ('kind', 'lit', ch)
    raise ParseError(f"bad TOP guard: {tok}")

class Transition:
    def __init__(self, state, in_guard, atop_guard, btop_guard, actions, next_state, halt=None, lineno=None, raw=None):
        self.state = state
        self.in_guard = in_guard
        self.atop_guard = atop_guard
        self.btop_guard = btop_guard
        self.actions = actions  # list of action tokens
        self.next_state = next_state
        self.halt = halt  # None | 'accept' | 'reject'
        self.lineno = lineno
        self.raw = raw

    def __repr__(self):
        return f"<Trans {self.state}:{self.in_guard},{self.atop_guard},{self.btop_guard} -> {self.actions} ; {self.next_state or self.halt}>"

def parse_line(line: str, lineno: int):
    # <state> : <IN>, A=<ATOP>, B=<BTOP> -> <actions> ; <nextState>
    # or with halt:* and no nextState
    # strip comments
    code = line.split('#', 1)[0].strip()
    if not code:
        return None
    if code.startswith('.start'):
        parts = code.split()
        if len(parts) != 2:
            raise ParseError(f".start directive expects one state name (line {lineno})")
        return ('start', parts[1])
    # split on ':'
    if ':' not in code:
        raise ParseError(f"missing ':' (line {lineno})")
    state, rest = code.split(':', 1)
    state = state.strip()
    rest = rest.strip()

    # split guards and arrow
    if '->' not in rest:
        raise ParseError(f"missing '->' (line {lineno})")
    guards_part, action_part = rest.split('->', 1)
    guards_part = guards_part.strip()
    action_part = action_part.strip()

    # guards: <IN>, A=<ATOP>, B=<BTOP>
    # Split by commas
    gparts = [g.strip() for g in guards_part.split(',')]
    if len(gparts) != 3:
        raise ParseError(f"guards must be '<IN>, A=<ATOP>, B=<BTOP>' (line {lineno})")
    in_tok = gparts[0]
    if not in_tok:
        raise ParseError(f"empty IN guard (line {lineno})")
    a_tok = gparts[1]
    b_tok = gparts[2]
    if not a_tok.startswith('A=') or not b_tok.startswith('B='):
        raise ParseError(f"second/third guard must be 'A=...' and 'B=...' (line {lineno})")

    in_guard = parse_guard_in(in_tok)
    atop_guard = parse_guard_top(a_tok[2:].strip(' '))
    btop_guard = parse_guard_top(b_tok[2:].strip(' '))

    # split actions and next
    next_state = None
    halt = None
    actions_str = action_part
    if ';' in action_part:
        act_str, next_str = action_part.split(';', 1)
        actions_str = act_str.strip()
        next_state = next_str.strip() or None
    actions = split_actions(actions_str) if actions_str else []

    # detect halt action
    for a in actions:
        if a.startswith('halt:'):
            h = a.split(':', 1)[1].strip()
            if h not in ('accept', 'reject'):
                raise ParseError(f"bad halt action '{a}' (line {lineno})")
            halt = h
    if halt and next_state:
        raise ParseError(f"halt action cannot have a next state (line {lineno})")

    return Transition(state, in_guard, atop_guard, btop_guard, actions, next_state, halt, lineno, raw=line.rstrip('\n'))

def guard_in_matches(in_guard, ptr, data):
    kind, val, ch = in_guard
    if val == 'eps':
        return True
    if val == 'eof':
        return ptr >= len(data)
    if val == 'any':
        return ptr < len(data)
    if val == 'lit':
        return ptr < len(data) and data[ptr] == ch
    return False

def guard_top_matches(top_guard, stack):
    kind, val, ch = top_guard
    if val == '*':
        return True
    if val == 'empty':
        return len(stack) == 0
    if val == 'lit':
        return len(stack) > 0 and stack[-1] == ch
    return False

def run(program, input_data, max_steps=1_000_000, trace=False, outstream=None):
    start_state = program['start']
    trans_by_state = program['transitions']

    state = start_state
    A = []
    B = []
    ptr = 0
    out = outstream if outstream is not None else sys.stdout
    last = None

    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeFault(f"Exceeded max steps {max_steps}")
        trs = trans_by_state.get(state, [])
        chosen = None
        for t in trs:
            if guard_in_matches(t.in_guard, ptr, input_data) and \
               guard_top_matches(t.atop_guard, A) and \
               guard_top_matches(t.btop_guard, B):
                chosen = t
                break
        if chosen is None:
            raise RuntimeFault(f"No transition matches in state '{state}' at ptr={ptr}, A_top={'empty' if not A else repr(A[-1])}, B_top={'empty' if not B else repr(B[-1])}")

        if trace:
            at = 'empty' if not A else repr(A[-1])
            bt = 'empty' if not B else repr(B[-1])
            preview = input_data[ptr:ptr+10].replace('\n', '\\n')
            sys.stderr.write(f"[{steps}] state={state} ptr={ptr} in='{preview}' A_top={at} B_top={bt}\n")
            sys.stderr.write(f"      rule: {chosen.raw}\n")

        # execute actions
        for act in chosen.actions:
            if not act:
                continue
            if act == 'read':
                if ptr >= len(input_data):
                    raise RuntimeFault("read at EOF")
                last = input_data[ptr]
                ptr += 1
            elif act.startswith('write:'):
                rhs = act[len('write:'):]
                if rhs == 'last':
                    if last is None:
                        raise RuntimeFault("write:last with empty last")
                    out.write(last)
                else:
                    s = unescape_string(rhs)
                    out.write(s)
            elif act.startswith('pushA:'):
                rhs = act[len('pushA:'):]
                if rhs == 'last':
                    if last is None:
                        raise RuntimeFault("pushA:last with empty last")
                    A.append(last)
                elif rhs.startswith(" '") or rhs.startswith("'") or rhs.startswith('"'):
                    # normalize to single-quoted form
                    if rhs[0] == "'":
                        ch = unescape_char(rhs[1:-1])
                    elif rhs.startswith(" '"):
                        ch = unescape_char(rhs.strip()[1:-1])
                    else:
                        # double-quoted illegal for char
                        raise ParseError(f"char literal must be single-quoted: {act}")
                    A.append(ch)
                elif rhs.startswith("lit:'"):
                    ch = parse_char_token(rhs)
                    A.append(ch)
                else:
                    # try single-quoted
                    rhs2 = rhs.strip()
                    if rhs2.startswith("'") and rhs2.endswith("'"):
                        ch = unescape_char(rhs2[1:-1])
                        A.append(ch)
                    else:
                        raise ParseError(f"bad pushA arg: {act}")
            elif act.startswith('pushB:'):
                rhs = act[len('pushB:'):]
                if rhs == 'last':
                    if last is None:
                        raise RuntimeFault("pushB:last with empty last")
                    B.append(last)
                else:
                    rhs2 = rhs.strip()
                    if rhs2.startswith("'") and rhs2.endswith("'"):
                        ch = unescape_char(rhs2[1:-1])
                        B.append(ch)
                    elif rhs2.startswith("lit:'"):
                        ch = parse_char_token(rhs2)
                        B.append(ch)
                    else:
                        raise ParseError(f"bad pushB arg: {act}")
            elif act == 'popA':
                if not A:
                    raise RuntimeFault("popA on empty stack")
                A.pop()
            elif act == 'popB':
                if not B:
                    raise RuntimeFault("popB on empty stack")
                B.pop()
            elif act == 'popA:last':
                if not A:
                    raise RuntimeFault("popA:last on empty stack")
                last = A.pop()
            elif act == 'popB:last':
                if not B:
                    raise RuntimeFault("popB:last on empty stack")
                last = B.pop()
            elif act == 'halt:accept':
                return 0
            elif act == 'halt:reject':
                return 1
            else:
                raise ParseError(f"unknown action: {act}")

        if chosen.halt:
            return 0 if chosen.halt == 'accept' else 1

        if chosen.next_state is None:
            raise RuntimeFault(f"No next state and no halt after actions in state '{state}' line {chosen.lineno}")
        state = chosen.next_state

def parse_program(text: str):
    start_state = 'start'
    trans_by_state = {}
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        parsed = parse_line(line, i)
        if parsed is None:
            continue
        if isinstance(parsed, tuple) and parsed[0] == 'start':
            start_state = parsed[1]
            continue
        t: Transition = parsed
        trans_by_state.setdefault(t.state, []).append(t)
    return {'start': start_state, 'transitions': trans_by_state}

def main():
    ap = argparse.ArgumentParser(description="FA2S interpreter (Finite Automaton with 2 Stacks)")
    ap.add_argument('program', help='path to .fa2s program file')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--input', dest='input_str', help='input string')
    group.add_argument('--infile', dest='input_file', help='path to input file')
    ap.add_argument('--max-steps', type=int, default=1_000_000)
    ap.add_argument('--trace', action='store_true')
    args = ap.parse_args()

    with open(args.program, 'r', encoding='utf-8') as f:
        prog_text = f.read()
    program = parse_program(prog_text)

    if args.input_str is not None:
        data = args.input_str
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = f.read()
    else:
        data = sys.stdin.read()

    try:
        rc = run(program, data, max_steps=args.max_steps, trace=args.trace, outstream=sys.stdout)
        # default: accept->0, reject->1; keep stdout as written
        sys.exit(rc)
    except (ParseError, RuntimeFault) as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(2)

if __name__ == '__main__':
    main()