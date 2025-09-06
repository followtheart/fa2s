#!/bin/bash
# Build and test script for FA2S project with MLIR frontend

set -e

PROJECT_ROOT="/home/jq/code/2s"
BUILD_DIR="$PROJECT_ROOT/build"

echo "FA2S Project Build and Test Script"
echo "=================================="

# Clean and create build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building the project..."
make -j$(nproc)

echo "Build completed successfully!"

# Test FA2S interpreter
echo ""
echo "Testing FA2S interpreter..."
if [ -f "./fa2s" ]; then
    echo "FA2S CLI tool built successfully"
    # Test with echo example
    if [ -f "../examples/echo.fa2s" ]; then
        echo "Testing echo example:"
        echo "hello world" | ./fa2s ../examples/echo.fa2s
    fi
else
    echo "Warning: FA2S CLI tool not found"
fi

# Test MLIR frontend (if available)
echo ""
echo "Testing MLIR frontend..."
if [ -f "./mlir/tools/fa2s-translate/fa2s-translate" ]; then
    echo "FA2S-translate tool built successfully"
    
    # Test translation
    echo "Translating test input..."
    ./mlir/tools/fa2s-translate/fa2s-translate ../mlir/test/input.fa2s -o ../mlir/test/output.mlir
    
    echo "Translation output:"
    cat ../mlir/test/output.mlir
else
    echo "Warning: fa2s-translate not found (MLIR not available)"
fi

echo ""
echo "All tests completed!"
