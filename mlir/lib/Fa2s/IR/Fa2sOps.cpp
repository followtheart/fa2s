#include "Fa2s/IR/Fa2sOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace fa2s;

#define GET_OP_CLASSES
#include "Fa2s/IR/Fa2sOps.cpp.inc"