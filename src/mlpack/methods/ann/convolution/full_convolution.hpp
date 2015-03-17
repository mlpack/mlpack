/**
 * @file full_convolution.hpp
 * @author Shangtong Zhang
 *
 * Implementation of the full convolution.
 */
#ifndef __MLPACK_METHODS_ANN_CONVOLUTION_FULL_CONVOLUTION_HPP
#define __MLPACK_METHODS_ANN_CONVOLUTION_FULL_CONVOLUTION_HPP

#include <mlpack/core.hpp>
#include "valid_convolution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

class FullConvolution
{
 public:
  
  template<typename MatType>
  static void conv(const MatType& input, const MatType& kernel, MatType& output)
  {
    if (input.n_elem > ValidConvolution::inputThreshold &&
        kernel.n_elem > ValidConvolution::kernelThreshold)
      convFFT(input, kernel, output);
    else
      convSimple(input, kernel, output);
  }
  
  // Simple appraoch to perform convolution
  template<typename MatType>
  static void convSimple(const MatType& input, const MatType& kernel, MatType& output)
  {
    size_t rowPadding = kernel.n_rows - 1;
    size_t colPadding = kernel.n_cols - 1;
    MatType extInput = arma::zeros<MatType> (input.n_rows + 2 * rowPadding,
                                             input.n_cols + 2 * colPadding);
    
    extInput(arma::span(rowPadding, rowPadding + input.n_rows - 1),
             arma::span(colPadding, colPadding + input.n_cols - 1)) = input;
    
    ValidConvolution::convSimple(extInput, kernel, output);
  }
  
  // Perform full convolution with FFT, which is faster when scale is large enough
  template<typename MatType>
  static void convFFT(const MatType& input, const MatType& kernel, MatType& output)
  {
    ValidConvolution::convFFTHelper(input, kernel, output);
    output = output(arma::span(kernel.n_rows - 1, output.n_rows - 1),
        arma::span(kernel.n_cols - 1, output.n_cols - 1));
  }
};

// rotate the kernel and perform full convolution
class RotatedKernelFullConvolution
{
 public:
  template<typename MatType>
  static void conv(const MatType& input, const MatType& kernel, MatType& output)
  {
    MatType rotatedKernel(kernel.n_rows, kernel.n_cols);
    for (size_t j = 0; j < kernel.n_cols; ++j)
    {
      for (size_t i = 0; i < kernel.n_rows; ++i)
      {
        rotatedKernel(i, j) = kernel(kernel.n_rows - 1 - i, kernel.n_cols - j - 1);
      }
    }
    FullConvolution::conv(input, rotatedKernel, output);
  }
  
};

}; // namespace ann
}; // namespace mlpack

#endif
