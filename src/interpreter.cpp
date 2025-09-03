#include "fa2s/interpreter.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fa2s {

// ---------- internal helpers (anonymous namespace) ----------

namespace {

inline std::string ltrim(std::string s) {
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return s.substr(i);
}
inline std::string rtrim(std::string s) {
    if (s.empty()) return s;
    std::size_t i = s.size();
    while (i > 0 && std::isspace(static_cast<unsigned char>(s[i - 1]))) --i;
    return s.substr(0, i);
}
inline std::string trim(const std::string& s) { return rtrim(ltrim(s)); }

inline bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
}
inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && std::equal(s.end() - suf.size(), s.end(), suf.begin());
}

std::string unescape_string(const std::string& token) {
    if (token.size() < 2 || token.front() != '"' || token.back() != '"')
        throw ParseError("bad string literal: " + token);
    std::string s = token.substr(1, token.size() - 2);
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\\') {
            if (++i >= s.size()) throw ParseError("bad string escape");
            char e = s[i];
            switch (e) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '0': out.push_back('\0'); break;
                default: out.push_back(e); break; // unknown -> literal
            }
        } else {
            out.push_back(c);
        }
    }
    return out;
}

char unescape_char_inner(const std::string& inner) {
    if (inner.empty()) throw ParseError("empty char literal");
    if (inner[0] == '\\') {
        if (inner.size() == 1) throw ParseError("bad escape in char literal");
        char e = inner[1];
        switch (e) {
            case 'n': return '\n';
            case 't': return '\t';
            case 'r': return '\r';
            case '\\': return '\\';
            case '\'': return '\'';
            case '"': return '"';
            case '0': return '\0';
            default: return e;
        }
    }
    return inner[0];
}

char parse_char_literal_token(const std::string& tok) {
    if (!starts_with(tok, "lit:'") || !ends_with(tok, "'"))
        throw ParseError("bad char token: " + tok);
    std::string inner = tok.substr(5, tok.size() - 6);
    return unescape_char_inner(inner);
}

std::vector<std::string> split_actions(const std::string& s) {
    std::vector<std::string> parts;
    std::string buf;
    for (std::size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '"') {
            buf.push_back(c);
            ++i;
            bool closed = false;
            while (i < s.size()) {
                char d = s[i];
                buf.push_back(d);
                if (d == '"' && !(i > 0 && s[i - 1] == '\\')) {
                    closed = true;
                    break;
                }
                ++i;
            }
            if (!closed) throw ParseError("unclosed string literal in actions");
        } else if (c == ',') {
            std::string t = trim(buf);
            if (!t.empty()) parts.push_back(t);
            buf.clear();
        } else {
            buf.push_back(c);
        }
    }
    std::string t = trim(buf);
    if (!t.empty()) parts.push_back(t);

    std::vector<std::string> out;
    out.reserve(parts.size());
    for (auto& p : parts) {
        auto tp = trim(p);
        if (!tp.empty()) out.push_back(std::move(tp));
    }
    return out;
}

InGuard parse_guard_in(const std::string& tok0) {
    std::string tok = trim(tok0);
    if (tok == "eps") return {InKind::EPS, 0};
    if (tok == "eof") return {InKind::EOFK, 0};
    if (tok == "any") return {InKind::ANY, 0};
    if (starts_with(tok, "lit:'")) {
        char c = parse_char_literal_token(tok);
        return {InKind::LIT, c};
    }
    throw ParseError("bad IN guard: " + tok);
}

TopGuard parse_guard_top(const std::string& tok0) {
    std::string tok = trim(tok0);
    if (tok == "*") return {TopKind::ANYTOP, 0};
    if (tok == "empty") return {TopKind::EMPTY, 0};
    if (starts_with(tok, "lit:'")) {
        char c = parse_char_literal_token(tok);
        return {TopKind::LIT, c};
    }
    throw ParseError("bad TOP guard: " + tok);
}

Transition parse_line_to_transition(const std::string& line, int lineno, std::optional<std::string>& start_state_opt) {
    std::string code = line;
    // strip comment
    std::size_t hashpos = code.find('#');
    if (hashpos != std::string::npos) code = code.substr(0, hashpos);
    code = trim(code);
    if (code.empty()) throw ParseError("EMPTY_LINE"); // sentinel for skip

    if (starts_with(code, ".start")) {
        std::istringstream iss(code);
        std::string dot, st;
        iss >> dot >> st;
        if (!iss || st.empty())
            throw ParseError(".start directive expects one state name (line " + std::to_string(lineno) + ")");
        start_state_opt = st;
        throw ParseError("EMPTY_LINE"); // handled directive
    }

    std::size_t colon = code.find(':');
    if (colon == std::string::npos) throw ParseError("missing ':' (line " + std::to_string(lineno) + ")");
    std::string state = trim(code.substr(0, colon));
    std::string rest = trim(code.substr(colon + 1));

    std::size_t arrow = rest.find("->");
    if (arrow == std::string::npos) throw ParseError("missing '->' (line " + std::to_string(lineno) + ")");
    std::string guards_part = trim(rest.substr(0, arrow));
    std::string action_part = trim(rest.substr(arrow + 2));

    // guards: <IN>, A=<ATOP>, B=<BTOP>
    std::vector<std::string> gparts;
    {
        std::string tmp; 
        for (char c : guards_part) {
            if (c == ',') { gparts.push_back(trim(tmp)); tmp.clear(); }
            else tmp.push_back(c);
        }
        gparts.push_back(trim(tmp));
    }
    if (gparts.size() != 3) throw ParseError("guards must be '<IN>, A=<ATOP>, B=<BTOP>' (line " + std::to_string(lineno) + ")");
    std::string in_tok = trim(gparts[0]);
    std::string a_tok = trim(gparts[1]);
    std::string b_tok = trim(gparts[2]);
    if (!starts_with(a_tok, "A=") || !starts_with(b_tok, "B="))
        throw ParseError("second/third guard must be 'A=...' and 'B=...' (line " + std::to_string(lineno) + ")");

    InGuard in = parse_guard_in(in_tok);
    TopGuard atop = parse_guard_top(trim(a_tok.substr(2)));
    TopGuard btop = parse_guard_top(trim(b_tok.substr(2)));

    std::string actions_str, next_state;
    if (action_part.find(';') != std::string::npos) {
        std::size_t semi = action_part.find(';');
        actions_str = trim(action_part.substr(0, semi));
        next_state = trim(action_part.substr(semi + 1));
    } else {
        actions_str = trim(action_part);
        next_state.clear();
    }
    std::vector<std::string> actions = actions_str.empty() ? std::vector<std::string>{} : split_actions(actions_str);

    std::string halt;
    for (auto& a : actions) {
        if (starts_with(a, "halt:")) {
            std::string h = trim(a.substr(5));
            if (h != "accept" && h != "reject")
                throw ParseError("bad halt action '" + a + "' (line " + std::to_string(lineno) + ")");
            halt = h;
        }
    }
    if (!halt.empty() && !next_state.empty())
        throw ParseError("halt action cannot have a next state (line " + std::to_string(lineno) + ")");

    Transition t{state, in, atop, btop, actions, next_state, halt, lineno, line};
    return t;
}

bool guard_in_matches(const InGuard& g, std::size_t ptr, const std::string& data) {
    switch (g.kind) {
        case InKind::EPS:  return true;
        case InKind::EOFK: return ptr >= data.size();
        case InKind::ANY:  return ptr < data.size();
        case InKind::LIT:  return ptr < data.size() && data[ptr] == g.ch;
    }
    return false;
}

bool guard_top_matches(const TopGuard& g, const std::vector<char>& stack) {
    switch (g.kind) {
        case TopKind::ANYTOP: return true;
        case TopKind::EMPTY:  return stack.empty();
        case TopKind::LIT:    return !stack.empty() && stack.back() == g.ch;
    }
    return false;
}

} // namespace

// ---------- Interpreter implementation ----------

Interpreter::Interpreter() {
    out_ = &std::cout;
}

void Interpreter::loadProgramFromString(const std::string& text) {
    std::istringstream iss(text);
    prog_ = parseProgram_(iss);
}

void Interpreter::loadProgram(std::istream& in) {
    prog_ = parseProgram_(in);
}

void Interpreter::setInput(std::string data) {
    input_ = std::move(data);
}

void Interpreter::setOutput(std::ostream* out) {
    out_ = out ? out : &std::cout;
}

void Interpreter::setTrace(bool enabled) {
    trace_ = enabled;
}

void Interpreter::setMaxSteps(std::size_t steps) {
    max_steps_ = steps;
}

void Interpreter::resetVM_() {
    ptr_ = 0;
    A_.clear(); B_.clear();
    has_last_ = false; last_ = 0;
    state_ = prog_.start;
    steps_executed_ = 0;
}

int Interpreter::run() {
    resetVM_();
    return runVM_();
}

int Interpreter::runVM_() {
    while (true) {
        if (++steps_executed_ > max_steps_) {
            throw RuntimeFault("Exceeded max steps " + std::to_string(max_steps_));
        }
        const auto it = prog_.by_state.find(state_);
        const std::vector<Transition>* trs = (it != prog_.by_state.end()) ? &it->second : nullptr;
        const Transition* chosen = nullptr;
        if (trs) {
            for (const auto& t : *trs) {
                if (guard_in_matches(t.in, ptr_, input_) &&
                    guard_top_matches(t.atop, A_) &&
                    guard_top_matches(t.btop, B_)) {
                    chosen = &t;
                    break;
                }
            }
        }
        if (!chosen) {
            std::string at = A_.empty() ? "empty" : std::string("'") + A_.back() + "'";
            std::string bt = B_.empty() ? "empty" : std::string("'") + B_.back() + "'";
            throw RuntimeFault("No transition matches in state '" + state_ +
                               "' at ptr=" + std::to_string(ptr_) +
                               ", A_top=" + at + ", B_top=" + bt);
        }

        if (trace_) {
            std::string preview = input_.substr(ptr_, 20);
            for (char& c : preview) if (c == '\n') c = ' ';
            std::cerr << "[" << steps_executed_ << "] state=" << state_
                      << " ptr=" << ptr_
                      << " in='" << preview << "' "
                      << "A_top=" << (A_.empty() ? std::string("empty") : std::string("'") + A_.back() + "'")
                      << " B_top=" << (B_.empty() ? std::string("empty") : std::string("'") + B_.back() + "'")
                      << "\n";
            std::cerr << "      rule: " << chosen->raw << "\n";
        }

        // Execute actions
        for (const std::string& act_raw : chosen->actions) {
            const std::string act = trim(act_raw);
            if (act.empty()) continue;

            if (act == "read") {
                if (ptr_ >= input_.size()) throw RuntimeFault("read at EOF");
                last_ = input_[ptr_++];
                has_last_ = true;
            } else if (starts_with(act, "write:")) {
                std::string rhs = trim(act.substr(6));
                if (rhs == "last") {
                    if (!has_last_) throw RuntimeFault("write:last with empty last");
                    out_->put(last_);
                } else {
                    std::string s = unescape_string(rhs);
                    (*out_) << s;
                }
            } else if (starts_with(act, "pushA:")) {
                std::string rhs = trim(act.substr(6));
                if (rhs == "last") {
                    if (!has_last_) throw RuntimeFault("pushA:last with empty last");
                    A_.push_back(last_);
                } else if (starts_with(rhs, "lit:'") && ends_with(rhs, "'")) {
                    A_.push_back(parse_char_literal_token(rhs));
                } else {
                    std::string t = trim(rhs);
                    if (t.size() >= 2 && t.front() == '\'' && t.back() == '\'') {
                        std::string inner = t.substr(1, t.size() - 2);
                        A_.push_back(unescape_char_inner(inner));
                    } else {
                        throw ParseError("bad pushA arg: " + act);
                    }
                }
            } else if (starts_with(act, "pushB:")) {
                std::string rhs = trim(act.substr(6));
                if (rhs == "last") {
                    if (!has_last_) throw RuntimeFault("pushB:last with empty last");
                    B_.push_back(last_);
                } else if (starts_with(rhs, "lit:'") && ends_with(rhs, "'")) {
                    B_.push_back(parse_char_literal_token(rhs));
                } else {
                    std::string t = trim(rhs);
                    if (t.size() >= 2 && t.front() == '\'' && t.back() == '\'') {
                        std::string inner = t.substr(1, t.size() - 2);
                        B_.push_back(unescape_char_inner(inner));
                    } else {
                        throw ParseError("bad pushB arg: " + act);
                    }
                }
            } else if (act == "popA") {
                if (A_.empty()) throw RuntimeFault("popA on empty stack");
                A_.pop_back();
            } else if (act == "popB") {
                if (B_.empty()) throw RuntimeFault("popB on empty stack");
                B_.pop_back();
            } else if (act == "popA:last") {
                if (A_.empty()) throw RuntimeFault("popA:last on empty stack");
                last_ = A_.back(); A_.pop_back(); has_last_ = true;
            } else if (act == "popB:last") {
                if (B_.empty()) throw RuntimeFault("popB:last on empty stack");
                last_ = B_.back(); B_.pop_back(); has_last_ = true;
            } else if (act == "halt:accept") {
                return 0;
            } else if (act == "halt:reject") {
                return 1;
            } else {
                throw ParseError("unknown action: " + act);
            }
        }

        if (!chosen->halt.empty()) {
            return chosen->halt == "accept" ? 0 : 1;
        }

        if (chosen->next_state.empty())
            throw RuntimeFault("No next state and no halt after actions in state '" + state_ +
                               "' line " + std::to_string(chosen->lineno));
        state_ = chosen->next_state;
    }
}

// static
Program Interpreter::parseProgram_(std::istream& in) {
    Program p;
    std::string line;
    int lineno = 0;
    std::optional<std::string> start_state_opt;
    while (std::getline(in, line)) {
        ++lineno;
        try {
            Transition t = parse_line_to_transition(line, lineno, start_state_opt);
            p.by_state[t.state].push_back(std::move(t));
        } catch (const ParseError& e) {
            if (std::string(e.what()) == std::string("EMPTY_LINE")) continue;
            throw;
        }
    }
    if (start_state_opt.has_value()) p.start = *start_state_opt;
    return p;
}

} // namespace fa2s