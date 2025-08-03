#!/bin/bash

# Build script for Issue #3964 reproduction test
# This script should be run on ARM64 system to test the fix

echo "=============================================="
echo "Issue #3964 Reproduction Test Build Script"
echo "=============================================="
echo "Testing: ARM64 OpenMP Convolution Segfault"
echo "Issue: https://github.com/mlpack/mlpack/issues/3964"
echo ""

# Check if we're on ARM64
echo "Checking system architecture..."
ARCH=$(uname -m)
echo "Architecture: $ARCH"

if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    echo "✓ Running on ARM64 - can reproduce the original issue"
else
    echo "⚠ Not running on ARM64 - original issue may not reproduce"
fi
echo ""

# Check OpenMP
echo "Checking OpenMP availability..."
if command -v gcc &> /dev/null; then
    if gcc -fopenmp -dumpversion &> /dev/null; then
        echo "✓ OpenMP support detected"
    else
        echo "✗ OpenMP not available - install libomp-dev"
        exit 1
    fi
else
    echo "✗ GCC not found"
    exit 1
fi
echo ""

# Create build directory
echo "Setting up build environment..."
mkdir -p build_issue_3964
cd build_issue_3964

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -f ../issue_3964_CMakeLists.txt \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_STANDARD=17

if [ $? -ne 0 ]; then
    echo "✗ CMake configuration failed"
    exit 1
fi

# Build the test
echo ""
echo "Building the reproduction test..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "✗ Build failed"
    exit 1
fi

echo ""
echo "✓ Build successful!"
echo ""
echo "To run the test:"
echo "  cd build_issue_3964"
echo "  ./issue_3964_test"
echo ""
echo "Expected results:"
echo "- Before fix: Segmentation fault in naive_convolution.hpp:91"
echo "- After fix: Training completes successfully"
echo ""
echo "The test will show detailed output about the crash location"
echo "and whether the fix is working correctly."
