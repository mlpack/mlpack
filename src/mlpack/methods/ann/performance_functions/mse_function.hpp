/**
 * @file mse_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the mean squared error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_MSE_FUNCTION_HPP
#define MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_MSE_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The mean squared error performance function measures the network's
 * performance according to the mean of squared errors.
 */
class MeanSquaredErrorFunction
{
  public:
  /**
   * Computes the mean squared error function.
   *
   * @param network Network type of FFN, CNN or RNN
   * @param target Target data.
   * @param error same as place holder
   * @return sum of squared errors.
   */
  template<typename DataType, typename... Tp>
  static double Error(const std::tuple<Tp...>& network,
                      const DataType& target, const DataType &error)
  {
    return Error(std::get<sizeof...(Tp) - 1>(network).OutputParameter(),
                 target, error);
  }

  /**
   * Computes the mean squared error function.
   *
   * @param input Input data.
   * @param target Target data.
   * @return mean of squared errors.
   */
  template<typename DataType>
  static double Error(const DataType& input, const DataType& target, const DataType&)
  {
    return arma::mean(arma::mean(arma::square(target - input)));
  }

}; // class MeanSquaredErrorFunction

} // namespace ann
} // namespace mlpack

#endif
