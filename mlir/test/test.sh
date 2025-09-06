#!/bin/bash
# Test script for FA2S MLIR frontend

set -e

echo "Testing FA2S MLIR Translation..."

# Check if fa2s-translate exists
if [ ! -f "./build/mlir/tools/fa2s-translate/fa2s-translate" ]; then
    echo "Error: fa2s-translate not found. Please build the project first."
    exit 1
fi

# Test translation
echo "Translating test input..."
./build/mlir/tools/fa2s-translate/fa2s-translate mlir/test/input.fa2s -o mlir/test/output.mlir

echo "Translation completed. Output:"
cat mlir/test/output.mlir

echo "Done."
