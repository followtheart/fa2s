//===- fa2s-translate.cpp - FA2S translation tool ------------------------===//
//
// This file implements the fa2s-translate tool, which converts FA2S source
// code to MLIR representation.
//
//===----------------------------------------------------------------------===//

#include "Fa2s/IR/Fa2sDialect.h"
#include "Fa2s/IR/Fa2sOps.h"
#include "Fa2s/Support/SourceMgr.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace fa2s;

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool>
    splitInputFile("split-input-file",
                   llvm::cl::desc("Split the input file into pieces and "
                                  "process each chunk independently"),
                   llvm::cl::init(false));

static llvm::cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      llvm::cl::desc("Check that emitted diagnostics match "
                                     "expected-* lines on the corresponding line"),
                      llvm::cl::init(false));

static llvm::cl::opt<bool>
    verifyPasses("verify-each",
                 llvm::cl::desc("Run the verifier after each transformation pass"),
                 llvm::cl::init(true));

static llvm::cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialect",
                              llvm::cl::desc("Allow operation with no registered dialects"),
                              llvm::cl::init(false));

/// Parse FA2S source and convert to MLIR
static OwningOpRef<ModuleOp> parseFA2S(MLIRContext &context, StringRef input) {
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  
  // Parse FA2S source using our lexer
  fa2s::Lexer lexer(input.str());
  
  // Create a function to hold the FA2S automaton
  auto func_type = builder.getFunctionType({}, {});
  auto func = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), "fa2s_main", func_type);
  func.setPrivate();
  
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);
  
  // Parse the FA2S program structure
  std::string startState = "start"; // default
  
  for (;;) {
    auto token = lexer.next();
    if (token.kind == Token::EOFToken)
      break;
    
    if (token.kind == Token::EOL)
      continue;
      
    if (token.kind == Token::Identifier) {
      if (token.text == ".start") {
        auto stateToken = lexer.next();
        if (stateToken.kind == Token::Identifier) {
          startState = stateToken.text;
        }
        continue;
      }
      
      // Parse state transition
      std::string stateName = token.text;
      
      // Expect ':'
      token = lexer.next();
      if (token.kind != Token::Symbol || token.text != ":") {
        llvm::errs() << "Expected ':' after state name\n";
        return nullptr;
      }
      
      // For now, create a simple state operation
      auto stateOp = builder.create<fa2s::StateOp>(
          builder.getUnknownLoc(), 
          builder.getStringAttr(stateName));
      
      // TODO: Parse guard conditions and actions
      // This is a simplified version - full parser would handle:
      // - Input guards (eps, eof, any, lit:'x')
      // - Stack guards (*, empty, lit:'x') 
      // - Actions (read, write, push, pop, halt)
      // - Next state transitions
    }
  }
  
  builder.create<func::ReturnOp>(builder.getUnknownLoc());
  module.getBody()->push_back(func);
  
  return module;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  
  // Register dialects
  MLIRContext context;
  context.getOrLoadDialect<fa2s::Fa2sDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "FA2S translation tool\n");
  
  // Set up input file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << argv[0] << ": could not open input file " << inputFilename
                 << ": " << error.message() << "\n";
    return 1;
  }
  
  // Parse FA2S source
  auto module = parseFA2S(context, fileOrErr.get()->getBuffer());
  if (!module)
    return 1;
  
  // Verify the module
  if (failed(verify(*module))) {
    llvm::errs() << "Generated MLIR is invalid\n";
    return 1;
  }
  
  // Set up output file
  std::error_code error;
  auto output = std::make_unique<llvm::ToolOutputFile>(outputFilename, error,
                                                       static_cast<llvm::sys::fs::OpenFlags>(0));
  if (error) {
    llvm::errs() << argv[0] << ": could not open output file " << outputFilename
                 << ": " << error.message() << "\n";
    return 1;
  }
  
  // Print the module
  module->print(output->os());
  output->keep();
  
  return 0;
}
