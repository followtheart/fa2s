# FA2S MLIR Frontend

This directory contains the MLIR (Multi-Level Intermediate Representation) frontend for the FA2S (Finite Automaton with 2 Stacks) language.

## Overview

FA2S is a minimalistic language based on the computational model of "finite automaton + two unbounded stacks (A, B)". This MLIR frontend provides:

- **FA2S Dialect**: Custom MLIR dialect for representing FA2S operations
- **Translation Tool**: `fa2s-translate` converts FA2S source to MLIR
- **Operations**: Support for states, transitions, guards, and actions

## Directory Structure

```
mlir/
├─ cmake/
│  └─ MLIRConfig.cmake          # MLIR configuration
├─ include/Fa2s/
│  ├─ Fa2sDialect.td           # Dialect TableGen definition
│  ├─ Fa2sOps.td              # Operations TableGen definition
│  ├─ IR/
│  │  ├─ Fa2sDialect.h        # Dialect header
│  │  └─ Fa2sOps.h           # Operations header
│  └─ Support/SourceMgr.h     # Source management utilities
├─ lib/Fa2s/
│  ├─ IR/
│  │  ├─ Fa2sDialect.cpp      # Dialect implementation
│  │  └─ Fa2sOps.cpp         # Operations implementation
│  └─ Support/SourceMgr.cpp   # Source management implementation
├─ tools/fa2s-translate/
│  ├─ CMakeLists.txt          # Build configuration
│  └─ fa2s-translate.cpp      # Translation tool
├─ test/
│  ├─ input.fa2s              # Test FA2S program
│  └─ expected.mlir           # Expected MLIR output
└─ README.md                  # This file
```

## FA2S Dialect Operations

### Core Operations

- `fa2s.state`: Represents a state in the finite automaton
- `fa2s.guard`: Defines conditions for state transitions
- `fa2s.actions`: Contains actions to execute during transitions
- `fa2s.transition`: Specifies next state after actions

### Guard Operations

- `fa2s.input_guard`: Guards based on input stream (eps, eof, any, lit)
- `fa2s.stack_guard`: Guards based on stack tops (*, empty, lit)

### Action Operations

- `fa2s.read`: Read character from input to register
- `fa2s.write`: Write string or register to output
- `fa2s.push`: Push character to stack A or B
- `fa2s.pop`: Pop character from stack A or B
- `fa2s.halt`: Terminate with accept/reject

## Building

The MLIR frontend is integrated with the main FA2S build system. From the project root:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make fa2s-translate
```

## Usage

### Translating FA2S to MLIR

```bash
./build/mlir/tools/fa2s-translate/fa2s-translate input.fa2s -o output.mlir
```

### Example

Input (`examples/echo.fa2s`):
```
.start start

start : eps, A=empty, B=empty -> ; loop
loop  : any, A=*, B=* -> read, write:last ; loop
loop  : eof, A=*, B=* -> halt:accept
```

Output MLIR:
```mlir
module {
  func.func private @fa2s_main() {
    fa2s.state "start"
    fa2s.state "loop" {
      fa2s.guard {
        fa2s.input_guard "any"
        fa2s.stack_guard "A" "*"
        fa2s.stack_guard "B" "*"
      }
      fa2s.actions {
        fa2s.read
        fa2s.write "last"
      }
      fa2s.transition "loop"
    }
    // ... more states
    return
  }
}
```

## Integration with LLVM/MLIR

This frontend requires LLVM/MLIR to be installed. The build system automatically discovers the installation through the standard CMake configuration files.

### Required MLIR Components

- MLIRDialect framework
- MLIR IR infrastructure
- MLIR parser/printer
- Function dialect (for wrapping FA2S programs)

## Testing

Run tests to verify the translation:

```bash
cd build
ctest -R fa2s-mlir
```

## Development

### Adding New Operations

1. Define the operation in `Fa2sOps.td`
2. Run TableGen to generate headers
3. Implement operation semantics in `Fa2sOps.cpp`
4. Update the parser in `fa2s-translate.cpp`

### Extending the Dialect

The FA2S dialect can be extended to support:
- Optimization passes for FA2S programs
- Lowering to standard dialects (SCF, MemRef, etc.)
- Integration with MLIR's transformation infrastructure

## Future Work

- **Passes**: Implement optimization passes for FA2S
- **Lowering**: Lower FA2S dialect to standard MLIR dialects
- **Verification**: Add comprehensive verification for FA2S semantics
- **JIT**: Integration with MLIR's JIT compilation for direct execution
