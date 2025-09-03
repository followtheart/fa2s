# FA2S Interpreter Library (C++17)

This refactors the single-file interpreter into a reusable library.

Highlights
- Library target: fa2s_interpreter
- Public API (namespace fa2s):
  - Exceptions: ParseError, RuntimeFault
  - Data types: Program, Transition, InGuard, TopGuard
  - Class: Interpreter
    - loadProgramFromString / loadProgram(istream&)
    - setInput(string), setOutput(ostream*), setTrace(bool), setMaxSteps(size_t)
    - run() -> int (0=accept, 1=reject; throws on error)
    - stepsExecuted() -> size_t
- CLI example target: fa2s (optional)

Build
- CMake
  - mkdir build && cd build
  - cmake .. -DCMAKE_BUILD_TYPE=Release
  - cmake --build .
- Run CLI
  - ./fa2s examples/echo.fa2s --input "hello"
  - ./fa2s examples/reverse.fa2s --input "abcd"
  - ./fa2s examples/balparen.fa2s --input "(())()"

Notes
- Semantics match the Python and the original C++ single-file version:
  - read at EOF, empty stack pop, or no matching transition -> RuntimeFault
  - Step limit default 1,000,000 (configurable)
  - Trace prints to stderr; program output to setOutput(ostream*) (default cout)