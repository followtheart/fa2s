//===- Fa2sDialect.td - fa2s dialect ----------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def Fa2s_Dialect : Dialect {
  let name = "fa2s";
  let cppNamespace = "::fa2s";
  let summary = "Finite Automaton with 2 Stacks dialect";
  let description = [{
    High-level IR for FA2S: states, transitions, actions, and guards.
  }];
}