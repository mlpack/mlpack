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

class MeanPooling {
  
 public:
  
  template<typename MatType>
  static void pooling(const MatType& input, double& output) {
    output = arma::mean(arma::mean(input));
  }
  
  template<typename MatType>
  static void unpooling(const MatType& input, const double& value,
                        MatType& output) {
    double new_value = value / input.n_elem;
    output = arma::zeros<MatType>(input.n_rows, input.n_cols);
    for (size_t j = 0; j < output.n_cols; ++j) {
      for (size_t i = 0; i < output.n_rows; ++i) {
        output(i,j) = new_value;
      }
    }
  }
  
 private:
};

}; // namespace ann
}; // namespace mlpack

#endif
