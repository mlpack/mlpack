/**
 * @file valid_convolution.hpp
 * @author Shangtong Zhang
 *
 * Implementation of the valid convolution.
 */
#ifndef __MLPACK_METHODS_ANN_CONVOLUTION_VALID_CONVOLUTION_HPP
#define __MLPACK_METHODS_ANN_CONVOLUTION_VALID_CONVOLUTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

class ValidConvolution
{
 public:
  
  // Threshold to select approach for convolution.
  static const size_t inputThreshold = 49e4;
  static const size_t kernelThreshold = 4e4;
  
  template<typename MatType>
  static void conv(const MatType& input, const MatType& kernel, MatType& output)
  {
    if (input.n_elem > inputThreshold && kernel.n_elem > kernelThreshold)
      convFFT(input, kernel, output);
    else
      convSimple(input, kernel, output);
  }
  
  // Simple appraoch to perform convolution
  template<typename MatType>
  static void convSimple(const MatType& input, const MatType& kernel, MatType& output)
  {
    output = arma::zeros<MatType>(input.n_rows - kernel.n_rows + 1,
                                  input.n_cols - kernel.n_cols + 1);
    
    for (size_t j = 0; j < output.n_cols; ++j)
    {
      for (size_t i = 0; i < output.n_rows; ++i)
      {
        double sum = 0;
        size_t leftInput = i;
        size_t topInput = j;
        for (size_t kj = 0; kj < kernel.n_cols; ++kj)
        {
          for (size_t ki = 0; ki < kernel.n_rows; ++ki)
          {
            sum += kernel(ki, kj) * input(leftInput + ki, topInput + kj);
          }
        }
        output(i, j) = sum;
      }
    }
  }
  
  template<typename MatType>
  static void convBlock(const MatType& input, const MatType& kernel, MatType& output)
  {
    output = arma::zeros<MatType>(input.n_rows - kernel.n_rows + 1,
                                  input.n_cols - kernel.n_cols + 1);
    for (size_t kj = 0; kj < kernel.n_cols; ++kj)
    {
      for (size_t ki = 0; ki < kernel.n_rows; ++ki)
      {
        output += kernel(ki, kj) * input(arma::span(ki, ki + output.n_rows - 1),
            arma::span(kj, kj + output.n_cols - 1));
      }
    }
    
  }
  
  // Perform valid convolution with FFT, which is faster when scale is large enough
  template<typename MatType>
  static void convFFT(const MatType& input, const MatType& kernel, MatType& output)
  {
    convFFTHelper(input, kernel, output);
    output = output(arma::span(2 * kernel.n_rows - 2, output.n_rows - kernel.n_rows),
        arma::span(2 * kernel.n_cols - 2, output.n_cols - kernel.n_cols));
  }
  
  // Perform convolution with FFT, which is faster when scale is large enough
  template<typename MatType>
  static void convFFTHelper(const MatType& input, const MatType& kernel, MatType& output)
  {
    size_t rowPadding = kernel.n_rows - 1;
    size_t colPadding = kernel.n_cols - 1;
    MatType extInput = arma::zeros<MatType> (input.n_rows + 2 * rowPadding,
                                             input.n_cols + 2 * colPadding);
    extInput(arma::span(rowPadding, rowPadding + input.n_rows - 1),
             arma::span(colPadding, colPadding + input.n_cols - 1)) = input;
    
    MatType extKernel = arma::zeros<MatType>(extInput.n_rows, extInput.n_cols);
    
    extKernel(arma::span(0, kernel.n_rows - 1),
        arma::span(0, kernel.n_cols - 1)) = kernel;
    
    output = arma::real(ifft2(arma::fft2(extInput) % arma::fft2(extKernel)));
  }
};

}; // namespace ann
}; // namespace mlpack

#endif
