# MLIRConfig.cmake - Configuration for MLIR in FA2S project
# This file is auto-discovered from LLVM/MLIR installation

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Include LLVM/MLIR definitions
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add LLVM/MLIR definitions
add_definitions(${LLVM_DEFINITIONS})

# LLVM/MLIR components we need
set(LLVM_LINK_COMPONENTS
  Core
  Support
)

# MLIR libraries we need
set(MLIR_LIBS
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRExport
  MLIRTransforms
    MLIRSupport              # mlir::Support 库
    MLIRTranslation          # mlir-translate 的支持库
    MLIRDialectUtils
    LLVMOption               # 让 cl::Option 工作
    LLVMSupport              # LLVM 基础支持库
)

# Helper function to configure MLIR target
function(configure_mlir_target target)
  target_link_libraries(${target} PRIVATE ${MLIR_LIBS})
  target_include_directories(${target} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
  )
endfunction()
