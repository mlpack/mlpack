/**
 * @file max_pooling.hpp
 * @author Shangtong Zhang
 *
 * Definition of the MaxPooling class, which implements max pooling.
 */
#ifndef __MLPACK_METHODS_ANN_POOLING_MAX_POOLING_HPP
#define __MLPACK_METHODS_ANN_POOLING_MAX_POOLING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

class MaxPooling
{
 public:
  template<typename MatType>
  static void pooling(const MatType& input, double& output)
  {
    output = input.max();
  }
  
  template<typename MatType>
  static void unpooling(const MatType& input, const double& value,
                        MatType& output)
  {
    arma::uword row = 0;
    arma::uword col = 0;
    output = arma::zeros<MatType>(input.n_rows, input.n_cols);
    input.max(row, col);
    output(row, col) = value;
  }
  
};

}; // namespace ann
}; // namespace mlpack

#endif
