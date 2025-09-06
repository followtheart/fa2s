#pragma once
#include <string>
#include <vector>
#include <optional>

namespace fa2s {
struct Location { int line = 1, col = 1; };

struct Token {
  enum Kind { Identifier, String, Char, Symbol, EOL, EOFToken } kind;
  std::string text; // 原始文本
  Location loc;
};

class Lexer {
public:
  explicit Lexer(std::string input);
  Token next();
private:
  std::string src; size_t i=0; int line=1, col=1;
  char peek() const; char get(); void skipSpaces();
  Token lexIdent(); Token lexString(); Token lexChar();
};

struct Guard {
  std::string input; // eps|eof|any|lit
  std::optional<unsigned char> inputLit;
  std::string a; std::optional<unsigned char> aLit;
  std::string b; std::optional<unsigned char> bLit;
};

struct Action { std::string kind; // read|write|pushA|pushB|popA|popB|popA:last|...
                 std::optional<std::string> sval; // string
                 std::optional<unsigned char> cval; // char
                 bool useLast=false; };

struct Transition {
  Guard guard;
  std::vector<Action> actions;
  std::optional<std::string> next; // state name; empty -> halt by action
};

struct State { std::string name; std::vector<Transition> trans; };

struct Program { std::string start="start"; std::vector<State> states; };

// 递归下降解析 FA2S 文法（按题述 BNF 的子集，满足 MVP）
class Parser {
public:
  explicit Parser(std::string input);
  Program parse();
private:
  Lexer lex; Token cur;
  void bump(); bool accept(const std::string &s);
  void expect(const std::string &s);
  std::string expectIdentifier();
  std::string expectKeyword();
  std::string expectString();
  unsigned char expectChar();
  Guard parseGuard();
  std::vector<Action> parseActions();
};
}