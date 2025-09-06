# FA2S translate tool
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
add_executable(fa2s-translate fa2s-translate.cpp)

# Include directories
target_include_directories(fa2s-translate PRIVATE
    ${CMAKE_SOURCE_DIR}/mlir/include
    ${CMAKE_BINARY_DIR}/mlir/include
)


# Link libraries
target_link_libraries(fa2s-translate PRIVATE
    # FA2S libraries
    -Wl,--whole-archive
    MLIRFa2s
    -Wl,--no-whole-archive
    
    # Essential MLIR libraries for basic functionality
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRDialect
    MLIRFuncDialect
    MLIRTransforms
    MLIRAnalysis
    
    # LLVM libraries
    LLVMSupport
    LLVMCore
    LLVMOption
    LLVMDemangle
    LLVMBinaryFormat
)

# Use MLIR's check_all_link_libraries to ensure we have everything
# mlir_check_all_link_libraries(fa2s-translate)

# Installation
install(TARGETS fa2s-translate RUNTIME DESTINATION bin)