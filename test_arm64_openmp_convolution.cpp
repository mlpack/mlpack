/**
 * @file test_arm64_openmp_convolution.cpp
 * @author GitHub Copilot
 *
 * Test for ARM64 OpenMP convolution segmentation fault fix.
 * This test specifically targets the issue described in mlpack issue #3964.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

using namespace mlpack;

/**
 * Test that reproduces the ARM64 + OpenMP segmentation fault issue.
 * This test creates a scenario similar to the one that caused the crash.
 */
int main()
{
  // Set the number of OpenMP threads to test parallel execution
#ifdef _OPENMP
  omp_set_num_threads(4);
  std::cout << "OpenMP enabled with " << omp_get_max_threads() << " threads." << std::endl;
#else
  std::cout << "OpenMP not available for this test." << std::endl;
#endif

  try
  {
    // Create a neural network with convolution layers similar to the failing case
    FFN<NegativeLogLikelihood, RandomInitialization> model;
    
    // Add convolution layer that was causing the segfault
    model.Add<Convolution>(
        /* maps */ 8,      // Number of output feature maps
        /* kernelW */ 3,   // Kernel width
        /* kernelH */ 3,   // Kernel height  
        /* strideW */ 1,   // Stride width
        /* strideH */ 1,   // Stride height
        /* padW */ 0,      // Padding width
        /* padH */ 0       // Padding height
    );
    
    model.Add<ReLU>();
    
    // Add another convolution layer to stress test
    model.Add<Convolution>(
        /* maps */ 16,     // Number of output feature maps
        /* kernelW */ 3,   // Kernel width
        /* kernelH */ 3,   // Kernel height  
        /* strideW */ 1,   // Stride width
        /* strideH */ 1,   // Stride height
        /* padW */ 0,      // Padding width
        /* padH */ 0       // Padding height
    );
    
    model.Add<ReLU>();
    
    // Add dense layers
    model.Add<Linear>(100);
    model.Add<ReLU>();
    model.Add<Linear>(10);
    model.Add<LogSoftMax>();

    // Create test data that would trigger the original crash
    // Using dimensions that would cause pointer arithmetic issues on ARM64
    arma::mat trainData = arma::randu<arma::mat>(28 * 28, 100);  // 100 samples of 28x28 images
    arma::mat trainLabels = arma::zeros<arma::mat>(10, 100);
    
    // Set random labels
    for (size_t i = 0; i < trainLabels.n_cols; ++i)
    {
      trainLabels(arma::randi(arma::distr_param(0, 9)), i) = 1;
    }

    std::cout << "Training data shape: " << trainData.n_rows << "x" << trainData.n_cols << std::endl;
    std::cout << "Training labels shape: " << trainLabels.n_rows << "x" << trainLabels.n_cols << std::endl;

    // Configure training parameters
    ens::Adam optimizer(0.001, 32, 0.9, 0.999, 1e-8, 0, 1e-5, false, true);
    model.Train(trainData, trainLabels, optimizer);

    std::cout << "SUCCESS: ARM64 OpenMP convolution test completed without segfault!" << std::endl;
    std::cout << "The fix for issue #3964 appears to be working correctly." << std::endl;
    
    // Test forward pass with the trained model
    arma::mat testData = arma::randu<arma::mat>(28 * 28, 10);
    arma::mat predictions;
    model.Predict(testData, predictions);
    
    std::cout << "Forward pass completed successfully with predictions shape: " 
              << predictions.n_rows << "x" << predictions.n_cols << std::endl;

    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "ERROR: Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << "ERROR: Test failed with unknown exception" << std::endl;
    return 1;
  }
}
