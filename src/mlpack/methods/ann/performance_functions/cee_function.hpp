/**
 * @file cee_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the cross-entropy error performance
 * function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_CEE_FUNCTION_HPP
#define MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_CEE_FUNCTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The cross-entropy error performance function measures the network's
 * performance according to the cross entropy errors. The log in the cross-
 * entropy take sinto account the closeness of a prediction and is a more
 * granular way to calculate the error.
 *
 * @tparam Layer The layer that is connected with the output layer.
 */
template<
    class Layer = LinearLayer< >
>
class CrossEntropyErrorFunction
{
 public:
  /**
   * Computes the cross-entropy error function..
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
   * Computes the cross-entropy error function.
   *
   * @param input Input data.
   * @param target Target data.
   * @return cross-entropy error.
   */
  template<typename DataType>
  static double Error(const DataType& input, const DataType& target, const DataType&)
  {
    if (LayerTraits<Layer>::IsBinary)
      return -arma::dot(arma::trunc_log(arma::abs(target - input)), target);

    return -arma::dot(arma::trunc_log(input), target);
  }

}; // class CrossEntropyErrorFunction

} // namespace ann
} // namespace mlpack

#endif
