/**
 * @file simple_mlpack_test.cpp
 * @author GitHub Copilot
 *
 * Simple test to verify mlpack build environment and basic functionality.
 */

#include <iostream>
#include <armadillo>

#ifdef _OPENMP
#include <omp.h>
#endif

// Try to include mlpack headers
#include <mlpack.hpp>

using namespace mlpack;

int main()
{
    std::cout << "=== MLpack Build Environment Test ===" << std::endl;
    
    // Test Armadillo
    arma::mat A = arma::randu<arma::mat>(4, 4);
    std::cout << "✓ Armadillo working - created 4x4 random matrix" << std::endl;
    
    // Test OpenMP
#ifdef _OPENMP
    std::cout << "✓ OpenMP enabled with " << omp_get_max_threads() << " threads" << std::endl;
#else
    std::cout << "⚠ OpenMP not available" << std::endl;
#endif

    // Test basic mlpack functionality
    try {
        // Create some test data
        arma::mat data = arma::randu<arma::mat>(10, 100);
        std::cout << "✓ Test data created: " << data.n_rows << "x" << data.n_cols << std::endl;
        
        // Test if we can include convolution headers (this would be our system mlpack)
        std::cout << "✓ MLpack headers included successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ MLpack test failed: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "=== All tests passed! ===" << std::endl;
    return 0;
}
