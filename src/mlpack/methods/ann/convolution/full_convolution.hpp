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
  static void conv(const MatType& input, const MatType& kernel,
                   MatType& output)
  {
    size_t rowPadding = kernel.n_rows - 1;
    size_t colPadding = kernel.n_cols - 1;
    MatType extInput = arma::zeros<MatType> (input.n_rows + 2 * rowPadding,
                                             input.n_cols + 2 * colPadding);
    
    extInput(arma::span(rowPadding, rowPadding + input.n_rows - 1),
             arma::span(colPadding, colPadding + input.n_cols - 1)) = input;
    
    ValidConvolution::conv(extInput, kernel, output);
  }
  
};

}; // namespace ann
}; // namespace mlpack

#endif
