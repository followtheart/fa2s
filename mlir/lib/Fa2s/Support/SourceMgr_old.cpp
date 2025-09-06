#include "Fa2s/Support/SourceMgr.h"

#include <cctype>
#include <stdexcept>

using namespace fa2s;

Lexer::Lexer(std::string input) : src(std::move(input)) {}
char Lexer::peek() const { return i < src.size() ? src[i] : 0; }
char Lexer::get() {
  char c = peek();
  if (c) {
    ++i;
    if (c == '\n') {
      ++line;
      col = 1;
    } else
      ++col;
  }
  return c;
}
void Lexer::skipSpaces() {
  for (;;) {
    char c = peek();
    if (c == '#') {
      while (c && c != '\n') {
        c = get();
      }
    } else if (c == ' ' || c == '\t' || c == '\r') {
      get();
    } else
      break;
  }
}

Token Lexer::next() {
  skipSpaces();
  Location L{line, col};
  char c = peek();
  if (!c) return {Token::EOFToken, "", L};
  if (c == '\n') {
    get();
    return {Token::EOL, "\n", L};
  }
  if (std::isalpha((unsigned char)c) || c == '_' || c == '.') return lexIdent();
  if (c == '\"') return lexString();
  if (c == '\'') return lexChar();
  // symbols
  get();
  std::string t;
  t.push_back(c);
  return {Token::Symbol, t, L};
}

Token Lexer::lexIdent() {
  Location L{line, col};
  std::string t;
  char c = peek();
  while (std::isalnum((unsigned char)c) || c == '_' || c == '.' || c == ':') {
    t.push_back(get());
    c = peek();
  }
  return {Token::Identifier, t, L};
}

Token Lexer::lexString() {
  Location L{line, col};
  std::string t;
  get();  // opening "
  for (char c = peek(); c; c = peek()) {
    if (c == '\\') {
      t.push_back(get());
      if (peek()) t.push_back(get());
    } else if (c == '\"') {
      get();
      break;
    } else {
      t.push_back(get());
    }
  }
  return {Token::String, t, L};
}

Token Lexer::lexChar() {
  Location L{line, col};
  get();  // '
  char out = 0;
  char c = peek();
  if (c == '\\') {
    get();
    char e = get();  // simple escapes subset
    switch (e) {
      case 'n':
        out = '\n';
        break;
      case 't':
        out = '\t';
        break;
      case 'r':
        out = '\r';
        break;
      case '\\':
        out = '\\';
        break;
      case '\'':
        out = '\'';
        break;
      default:
        out = e;
    }
  } else {
    out = get();
  }
  if (peek() != '\'') throw std::runtime_error("unterminated char literal");
  get();
  std::string t;
  t.push_back(out);
  return {Token::Char, t, L};
}

Parser::Parser(std::string input) : lex(std::move(input)) { cur = lex.next(); }
// B=stack_guard
if (cur.kind == Token::Identifier && cur.text == "B") {
  bump();
} else
  throw std::runtime_error("expected 'B' in guard");
expect("=");
if (cur.kind == Token::Identifier) {
  if (cur.text == "*" || cur.text == "empty") {
    g.b = cur.text;
    bump();
  } else if (cur.text.rfind("lit:", 0) == 0) {
    auto litstr = cur.text.substr(4);
    if (litstr.size() >= 2 && litstr.front() == "'"[0]) {
      g.b = "lit";
      g.bLit = (unsigned char)litstr[1];
      bump();
    } else {
      throw std::runtime_error("bad B lit:");
    }
  } else {
    throw std::runtime_error("unknown B guard: " + cur.text);
  }
} else if (cur.kind == Token::Symbol && cur.text == "*") {
  g.b = "*";
  bump();
} else
  throw std::runtime_error("expected B guard");

return g;
}

// Parse action list: action (',' action)*
std::vector<Action> Parser::parseActions() {
  std::vector<Action> actions;
  for (;;) {
    if (cur.kind == Token::Identifier) {
      std::string id = cur.text;
      if (id == "read") {
        Action a;
        a.kind = "read";
        actions.push_back(a);
        bump();
      } else if (id == "write") {
        bump();
        expect(":");
        if (cur.kind == Token::String) {
          Action a;
          a.kind = "write";
          a.sval = cur.text;
          actions.push_back(a);
          bump();
        } else if (cur.kind == Token::Identifier && cur.text == "last") {
          Action a;
          a.kind = "write:last";
          actions.push_back(a);
          bump();
        } else
          throw std::runtime_error("expected string or last after write:");
      } else if (id.rfind("pushA", 0) == 0 || id.rfind("pushB", 0) == 0) {
        std::string base = id;
        bump();
        expect(":");
        if (cur.kind == Token::Char) {
          Action a;
          a.kind = (base.find('A') != std::string::npos) ? "pushA" : "pushB";
          a.cval = (unsigned char)cur.text[0];
          actions.push_back(a);
          bump();
        } else if (cur.kind == Token::Identifier && cur.text == "last") {
          Action a;
          a.kind = (base.find('A') != std::string::npos) ? "pushA" : "pushB";
          a.useLast = true;
          actions.push_back(a);
          bump();
        } else
          throw std::runtime_error("expected char or last after push:");
      } else if (id == "popA" || id == "popB") {
        Action a;
        a.kind = id;
        actions.push_back(a);
        bump();
      } else if (id == "popA:last" || id == "popB:last") {
        Action a;
        a.kind = id;
        actions.push_back(a);
        bump();
      } else if (id.rfind("halt:", 0) == 0) {
        Action a;
        a.kind = id;
        actions.push_back(a);
        bump();
      } else {
        throw std::runtime_error(std::string("unknown action: ") + id);
      }
    } else {
      throw std::runtime_error("expected action");
    }
    if (cur.kind == Token::Symbol && cur.text == ",") {
      bump();
      continue;
    }
    break;
  }
  return actions;
}

// Top-level parse: program -> (.start state? EOL?)* state*
Program Parser::parse() {
  Program P;
  while (!isEOLorEOF(cur)) {
    if (cur.kind == Token::EOL) {
      bump();
      continue;
    }
    if (cur.kind == Token::Identifier && cur.text == ".start") {
      bump();
      auto name = expectIdentifier();
      P.start = name;
      if (cur.kind == Token::EOL) bump();
      continue;
    }
    if (cur.kind == Token::Identifier) {
      State S;
      S.name = expectIdentifier();
      expect(":");
      Guard g = parseGuard();
      expect("->");
      auto acts = parseActions();
      expect(";");
      // next token: state name or nothing
      if (cur.kind == Token::Identifier) {
        std::string nxt = cur.text;
        bump();
        Transition T;
        T.guard = g;
        T.actions = acts;
        if (nxt == "halt:accept" || nxt == "halt:reject")
          T.next = std::nullopt;
        else
          T.next = nxt;
        S.trans.push_back(T);
      } else {
        Transition T;
        T.guard = g;
        T.actions = acts;
        T.next = std::nullopt;
        S.trans.push_back(T);
      }
      if (cur.kind == Token::EOL) bump();
      auto it =
          std::find_if(P.states.begin(), P.states.end(),
                       [&](const State &st) { return st.name == S.name; });
      if (it == P.states.end())
        P.states.push_back(S);
      else
        it->trans.insert(it->trans.end(), S.trans.begin(), S.trans.end());
      continue;
    }
    bump();
  }
  return P;
}