#ifndef FA2S_INTERPRETER_HPP
#define FA2S_INTERPRETER_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <optional>
#include <iosfwd>

namespace fa2s {

// Exceptions
struct ParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct RuntimeFault : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Guards and transition model
enum class InKind { EPS, EOFK, ANY, LIT };
enum class TopKind { ANYTOP, EMPTY, LIT };

struct InGuard {
    InKind kind{};
    char ch{};
};

struct TopGuard {
    TopKind kind{};
    char ch{};
};

struct Transition {
    std::string state;
    InGuard in;
    TopGuard atop;
    TopGuard btop;
    std::vector<std::string> actions; // raw action tokens
    std::string next_state;           // empty if none
    std::string halt;                 // "", "accept", "reject"
    int lineno{};
    std::string raw;                  // original line for tracing
};

struct Program {
    std::string start = "start";
    std::unordered_map<std::string, std::vector<Transition>> by_state;
};

// Interpreter
class Interpreter {
public:
    Interpreter();

    // Load/parse program
    void loadProgramFromString(const std::string& text);
    void loadProgram(std::istream& in);
    const Program& program() const noexcept { return prog_; }

    // IO and config
    void setInput(std::string data);
    void setOutput(std::ostream* out);     // default: &std::cout
    void setTrace(bool enabled);           // default: false (trace to std::cerr)
    void setMaxSteps(std::size_t steps);   // default: 1'000'000

    // Execute: returns 0 (accept) or 1 (reject). Throws on error.
    int run();

    // Introspection
    std::size_t stepsExecuted() const noexcept { return steps_executed_; }

private:
    Program prog_;

    // VM state
    std::string input_;
    std::size_t ptr_{0};
    std::vector<char> A_, B_;
    bool has_last_{false};
    char last_{0};
    std::string state_;

    bool trace_{false};
    std::size_t max_steps_{1'000'000};
    std::ostream* out_{nullptr};
    std::size_t steps_executed_{0};

    void resetVM_();
    int runVM_();

    // Parsing helpers (implemented in .cpp)
    static Program parseProgram_(std::istream& in);
};

} // namespace fa2s

#endif // FA2S_INTERPRETER_HPP