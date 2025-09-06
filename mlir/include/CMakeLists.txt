# TableGen files for FA2S dialect

# Create output directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR)

# Generate Dialect files
set(LLVM_TARGET_DEFINITIONS Fa2s/Fa2sDialect.td)  
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sDialect.h.inc
  COMMAND mlir-tblgen -gen-dialect-decls -dialect=fa2s 
          -I ${MLIR_INCLUDE_DIRS} -I ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sDialect.td
          -o ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sDialect.h.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sDialect.td
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sDialect.cpp.inc
  COMMAND mlir-tblgen -gen-dialect-defs -dialect=fa2s 
          -I ${MLIR_INCLUDE_DIRS} -I ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sDialect.td
          -o ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sDialect.cpp.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sDialect.td
)

# Generate Operations files  
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sOps.h.inc
  COMMAND mlir-tblgen -gen-op-decls
          -I ${MLIR_INCLUDE_DIRS} -I ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sOps.td
          -o ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sOps.h.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sOps.td ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sDialect.td
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sOps.cpp.inc
  COMMAND mlir-tblgen -gen-op-defs
          -I ${MLIR_INCLUDE_DIRS} -I ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sOps.td
          -o ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sOps.cpp.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sOps.td ${CMAKE_CURRENT_SOURCE_DIR}/Fa2s/Fa2sDialect.td
)

# Create custom targets
add_custom_target(Fa2sDialectIncGen DEPENDS 
  ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sDialect.h.inc 
  ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sDialect.cpp.inc)

add_custom_target(Fa2sOpsIncGen DEPENDS 
  ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sOps.h.inc 
  ${CMAKE_CURRENT_BINARY_DIR}/Fa2s/IR/Fa2sOps.cpp.inc)

# Combined target
add_custom_target(MLIRFa2sIncGen DEPENDS Fa2sOpsIncGen Fa2sDialectIncGen)