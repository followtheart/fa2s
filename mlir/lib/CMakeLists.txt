# FA2S MLIR library

add_library(MLIRFa2s
  Fa2s/IR/Fa2sDialect.cpp
  Fa2s/IR/Fa2sOps.cpp
  Fa2s/Support/SourceMgr.cpp
)

target_include_directories(MLIRFa2s PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

add_dependencies(MLIRFa2s MLIRFa2sIncGen)

target_link_libraries(MLIRFa2s PUBLIC
  MLIRIR
  MLIRSupport
  LLVMSupport
)