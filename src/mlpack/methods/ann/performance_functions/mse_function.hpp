/**
 * @file mse_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the mean squared error performance function.
 */
#ifndef __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_MSE_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_MSE_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The mean squared error performance function measures the network's
 * performance according to the mean of squared errors.
 *
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template<typename VecType = arma::colvec>
class MeanSquaredErrorFunction
{
  public:
  /**
   * Computes the mean squared error function.
   *
   * @param input Input data.
   * @param target Target data.
   * @return mean of squared errors.
   */
  static double error(const VecType& input, const VecType& target)
  {
    return arma::mean(arma::square(target - input));
  }

}; // class MeanSquaredErrorFunction

}; // namespace ann
}; // namespace mlpack

#endif
