#include "Fa2s/IR/Fa2sDialect.h"
#include "Fa2s/IR/Fa2sOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace fa2s;

// Include the auto-generated dialect definitions
#include "Fa2s/IR/Fa2sDialect.cpp.inc"

void Fa2sDialect::initialize() {
  addOperations<
    ProgramOp,
    StateOp,
    TransitionOp,
    ReadOp,
    WriteOp,
    PushOp,
    PopOp,
    HaltOp>();
}