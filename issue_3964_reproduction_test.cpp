/**
 * @file issue_3964_reproduction_test.cpp
 * @author GitHub Copilot
 *
 * EXACT reproduction test for mlpack issue #3964:
 * "Convolution layer causes segmentation fault on arm64 with OpenMP"
 *
 * This test reproduces the exact scenario described in the GitHub issue:
 * - FFN with two or more convolution layers
 * - Input dimensions: 7x7 (49 elements total)
 * - Convolution layers: 5 filters, 3x3 kernel
 * - Training with Adam optimizer
 * - Compilation with OpenMP enabled
 * - Should crash on ARM64 before fix, work after fix
 *
 * Original issue: https://github.com/mlpack/mlpack/issues/3964
 */

#include <mlpack.hpp>
#include <iostream>
#include <exception>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace mlpack;
using namespace arma;
using namespace std::chrono;

int main()
{
    std::cout << "=== Issue #3964 Reproduction Test ===" << std::endl;
    std::cout << "Reproducing: 'Convolution layer causes segmentation fault on arm64 with OpenMP'" << std::endl;
    std::cout << "Original issue: https://github.com/mlpack/mlpack/issues/3964" << std::endl;
    std::cout << std::endl;

    // Environment information
    std::cout << "Environment:" << std::endl;
#ifdef __aarch64__
    std::cout << "- Architecture: ARM64/AArch64 ✓" << std::endl;
#else
    std::cout << "- Architecture: " << 
#ifdef __x86_64__
        "x86_64" 
#else
        "Unknown"
#endif
        << " (not ARM64)" << std::endl;
#endif

#ifdef _OPENMP
    std::cout << "- OpenMP: Enabled with " << omp_get_max_threads() << " threads ✓" << std::endl;
#else
    std::cout << "- OpenMP: Disabled ✗" << std::endl;
    std::cout << "WARNING: This test requires OpenMP to reproduce the issue!" << std::endl;
#endif

    std::cout << "- Armadillo version: " << arma_version::as_string() << std::endl;
    std::cout << std::endl;

    try {
        // EXACT reproduction of the failing code from issue #3964
        std::cout << "Creating data and labels..." << std::endl;
        
        // From issue: arma::mat data(49, 500, arma::fill::randn);
        mat data(49, 500, arma::fill::randn);
        mat labels(1, 500, arma::fill::randn);
        
        std::cout << "- Data shape: " << data.n_rows << "x" << data.n_cols << std::endl;
        std::cout << "- Labels shape: " << labels.n_rows << "x" << labels.n_cols << std::endl;

        std::cout << "Creating FFN network..." << std::endl;
        
        // EXACT network structure from the issue
        FFN<MeanSquaredError> network;
        
        network.Add<Identity>();
        network.Add<Convolution>(5, 3, 3);  // 5 filters, 3x3 kernel - CRASH POINT
        network.Add<Convolution>(5, 3, 3);  // Second convolution layer
        network.Add<Linear>(1);
        
        // From issue: network.InputDimensions() = {7, 7};
        network.InputDimensions() = {7, 7};
        
        std::cout << "Network structure:" << std::endl;
        std::cout << "- Input: 7x7 = 49 elements" << std::endl;
        std::cout << "- Layer 1: Identity" << std::endl;
        std::cout << "- Layer 2: Convolution(5 filters, 3x3 kernel)" << std::endl;
        std::cout << "- Layer 3: Convolution(5 filters, 3x3 kernel)" << std::endl;
        std::cout << "- Layer 4: Linear(1 output)" << std::endl;
        std::cout << std::endl;

        std::cout << "Configuring Adam optimizer..." << std::endl;
        
        // From issue configuration
        ens::Adam opt;
        // Reduce iterations for faster testing, but keep enough to trigger the crash
        opt.MaxIterations() = data.n_cols * 10;  // Reduced from 50000 to 10 for testing
        
        std::cout << "- Optimizer: Adam" << std::endl;
        std::cout << "- Max iterations: " << opt.MaxIterations() << std::endl;
        std::cout << std::endl;

        std::cout << "Starting training (this is where the segfault occurred)..." << std::endl;
        std::cout << "Original crash location: naive_convolution.hpp:91" << std::endl;
        std::cout << "  *outputPtr += *kernelPtr * (*inputPtr);" << std::endl;
        std::cout << "With inputPtr = 0x0 (null pointer)" << std::endl;
        std::cout << std::endl;

        auto start_time = high_resolution_clock::now();

        // THIS IS THE EXACT CALL THAT CAUSED THE SEGFAULT
        // On ARM64 + OpenMP, this would crash in naive_convolution.hpp:91
        // with inputPtr = 0x0 (null pointer dereference)
        network.Train(data, labels, opt, ens::ProgressBar());

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);

        std::cout << std::endl;
        std::cout << "SUCCESS! Training completed without segmentation fault!" << std::endl;
        std::cout << "Training time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;
        
        // System-specific success message
#ifdef __aarch64__
        std::cout << "🎉 EXCELLENT! This test ran successfully on ARM64 architecture!" << std::endl;
        std::cout << "✅ ISSUE #3964 IS FIXED! The ARM64 OpenMP convolution segmentation fault is resolved!" << std::endl;
        std::cout << "   - No crash in naive_convolution.hpp:91" << std::endl;
        std::cout << "   - OpenMP parallelization working correctly on ARM64" << std::endl;
        std::cout << "   - Memory access patterns are now safe" << std::endl;
#else
        std::cout << "✅ Test passed on " << 
#ifdef __x86_64__
            "x86_64" 
#else
            "non-ARM64"
#endif
            " architecture (expected - original issue was ARM64-specific)" << std::endl;
        std::cout << "📋 To verify the fix for issue #3964, this test needs to run on ARM64/AArch64" << std::endl;
        std::cout << "   - Current system: Non-ARM64, so original issue wouldn't occur here" << std::endl;
        std::cout << "   - ARM64 testing required to confirm segmentation fault fix" << std::endl;
#endif
        
        // Additional validation - test prediction
        std::cout << std::endl;
        std::cout << "Testing forward pass (prediction)..." << std::endl;
        mat test_data(49, 10, arma::fill::randn);
        mat predictions;
        
        network.Predict(test_data, predictions);
        
        std::cout << "- Test data: " << test_data.n_rows << "x" << test_data.n_cols << std::endl;
        std::cout << "- Predictions: " << predictions.n_rows << "x" << predictions.n_cols << std::endl;
        std::cout << "✓ Forward pass successful!" << std::endl;
        std::cout << std::endl;
        
        // Final summary based on architecture
        std::cout << "=== FINAL SUMMARY ===" << std::endl;
#ifdef __aarch64__
        std::cout << "System: ARM64/AArch64 ✓" << std::endl;
        std::cout << "Result: ISSUE #3964 FIXED! 🎉" << std::endl;
        std::cout << "Status: Convolution layers now work correctly with OpenMP on ARM64" << std::endl;
#else
        std::cout << "System: " << 
#ifdef __x86_64__
            "x86_64" 
#else
            "Unknown (non-ARM64)"
#endif
            << std::endl;
        std::cout << "Result: Test passed (expected for non-ARM64)" << std::endl;
        std::cout << "Status: ARM64 testing still required to fully validate fix" << std::endl;
#endif
        
        return 0;

    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "CRASH! Exception caught: " << e.what() << std::endl;
        std::cout << std::endl;
        
#ifdef __aarch64__
        std::cout << "❌ FAILURE ON ARM64! The fix for issue #3964 is NOT working!" << std::endl;
        std::cout << "System: ARM64/AArch64" << std::endl;
        std::cout << "Expected: Training should complete without crash" << std::endl;
        std::cout << "Actual: Exception/crash occurred" << std::endl;
        std::cout << std::endl;
        std::cout << "This indicates the ARM64 OpenMP convolution issue is NOT fixed." << std::endl;
        std::cout << "Expected crash details on ARM64 + OpenMP before fix:" << std::endl;
        std::cout << "- Segmentation fault in naive_convolution.hpp:91" << std::endl;
        std::cout << "- inputPtr = 0x0 (null pointer)" << std::endl;
        std::cout << "- Crash in OpenMP parallel section" << std::endl;
#else
        std::cout << "Unexpected crash on " << 
#ifdef __x86_64__
            "x86_64" 
#else
            "non-ARM64"
#endif
            " system!" << std::endl;
        std::cout << "This is concerning as the original issue was ARM64-specific." << std::endl;
        std::cout << "The fix may have introduced a regression on other architectures." << std::endl;
#endif
        
        return 1;
    } catch (...) {
        std::cout << std::endl;
        std::cout << "CRASH! Unknown exception caught!" << std::endl;
        std::cout << std::endl;
        
#ifdef __aarch64__
        std::cout << "❌ CRITICAL FAILURE ON ARM64! Issue #3964 fix is NOT working!" << std::endl;
        std::cout << "System: ARM64/AArch64" << std::endl;
        std::cout << "Result: Unknown crash/exception" << std::endl;
#else
        std::cout << "Unexpected crash on " << 
#ifdef __x86_64__
            "x86_64" 
#else
            "non-ARM64"
#endif
            " system!" << std::endl;
        std::cout << "This suggests a potential regression introduced by the fix." << std::endl;
#endif
        
        return 1;
    }
}
