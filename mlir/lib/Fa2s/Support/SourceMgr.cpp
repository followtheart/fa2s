#include "Fa2s/Support/SourceMgr.h"

#include <cctype>
#include <stdexcept>
#include <algorithm>

using namespace fa2s;

Lexer::Lexer(std::string input) : src(std::move(input)) {}

char Lexer::peek() const { 
  return i < src.size() ? src[i] : 0; 
}

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
  if (std::isalpha((unsigned char)c) || c == '_' || c == '.') 
    return lexIdent();
  if (c == '\"') 
    return lexString();
  if (c == '\'') 
    return lexChar();
  
  // symbols
  get();
  std::string t;
  t.push_back(c);
  return {Token::Symbol, t, L};
}

Token Lexer::lexIdent() {
  Location L{line, col};
  std::string t;
  for (;;) {
    char c = peek();
    if (std::isalnum((unsigned char)c) || c == '_' || c == '.' || c == ':') {
      t.push_back(c);
      get();
    } else
      break;
  }
  return {Token::Identifier, t, L};
}

Token Lexer::lexString() {
  Location L{line, col};
  get(); // consume "
  std::string t;
  for (;;) {
    char c = peek();
    if (!c || c == '\n') 
      throw std::runtime_error("unterminated string");
    if (c == '\"') {
      get();
      break;
    }
    if (c == '\\') {
      get();
      char esc = get();
      if (esc == 'n') t.push_back('\n');
      else if (esc == 't') t.push_back('\t');
      else if (esc == 'r') t.push_back('\r');
      else if (esc == '\\') t.push_back('\\');
      else if (esc == '\"') t.push_back('\"');
      else t.push_back(esc);
    } else {
      t.push_back(c);
      get();
    }
  }
  return {Token::String, t, L};
}

Token Lexer::lexChar() {
  Location L{line, col};
  get(); // consume '
  char out = 0;
  char c = peek();
  if (!c || c == '\n') 
    throw std::runtime_error("unterminated char");
  if (c == '\\') {
    get();
    char esc = get();
    if (esc == 'n') out = '\n';
    else if (esc == 't') out = '\t';
    else if (esc == 'r') out = '\r';
    else if (esc == '\\') out = '\\';
    else if (esc == '\'') out = '\'';
    else out = esc;
  } else {
    out = c;
    get();
  }
  if (get() != '\'') 
    throw std::runtime_error("expected ' after char");
  
  std::string t;
  t.push_back(out);
  return {Token::Char, t, L};
}

Parser::Parser(std::string input) : lex(std::move(input)) { 
  cur = lex.next(); 
}

void Parser::bump() { 
  cur = lex.next(); 
}

void Parser::expect(const std::string &s) {
  if (cur.text != s)
    throw std::runtime_error("expected " + s + " got " + cur.text);
  bump();
}

std::string Parser::expectIdentifier() {
  if (cur.kind != Token::Identifier)
    throw std::runtime_error("expected identifier");
  std::string s = cur.text;
  bump();
  return s;
}

// Simplified program parser for now
Program Parser::parse() {
  Program P;
  P.start = "start"; // default
  
  while (cur.kind != Token::EOFToken) {
    if (cur.kind == Token::EOL || cur.kind == Token::EOFToken) {
      bump();
      continue;
    }
    
    // Skip other tokens for now - this is a minimal implementation
    bump();
  }
  
  return P;
}
