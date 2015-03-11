/**
 * @file mean_pooling.hpp
 * @author Shangtong Zhang
 *
 * Definition of the MeanPooling class, which implements mean pooling.
 */
#ifndef __MLPACK_METHODS_ANN_POOLING_MEAN_POOLING_HPP
#define __MLPACK_METHODS_ANN_POOLING_MEAN_POOLING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

class MeanPooling
{
 public:
  template<typename MatType>
  static void pooling(const MatType& input, double& output)
  {
    output = arma::mean(arma::mean(input));
  }
  
  template<typename MatType>
  static void unpooling(const MatType& input, const double& value,
                        MatType& output)
  {
    output = MatType(input.n_rows, input.n_cols);
    output.fill(value / input.n_elem);
  }
};

}; // namespace ann
}; // namespace mlpack

#endif
