#pragma once
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

// Forward declarations
#include "Fa2s/IR/Fa2sOps.h.inc"

#define GET_OP_CLASSES
#include "Fa2s/IR/Fa2sOps.h.inc"