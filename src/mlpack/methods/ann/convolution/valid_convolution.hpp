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

class ValidConvolution {
 public:
  
  template<typename MatType>
  static void conv(const MatType& input, const MatType& kernel,
                   MatType& output) {
    
    output = arma::zeros<MatType>(input.n_rows - kernel.n_rows + 1,
                                  input.n_cols - kernel.n_cols  +1);
    
    for (size_t j = 0; j < output.n_cols; ++j) {
      for (size_t i = 0; i < output.n_rows; ++i) {
        
        double sum = 0;
        size_t leftInput = i;
        size_t topInput = j;
        for (size_t kj = 0; kj < kernel.n_cols; ++kj) {
          for (size_t ki = 0; ki < kernel.n_rows; ++ki) {
            sum += kernel(ki, kj) * input(leftInput + ki, topInput + kj);
          }
        }
        output(i, j) = sum;
        
      }
    }
  }
};

}; // namespace ann
}; // namespace mlpack

#endif
