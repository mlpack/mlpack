/**
 * @file cee_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the cross-entropy error performance
 * function.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_CEE_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_CEE_FUNCTION_HPP

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
   * Computes the cross-entropy error function.
   *
   * @param input Input data.
   * @param target Target data.
   * @return cross-entropy error.
   */
  template<typename DataType>
  static double Error(const DataType& input, const DataType& target)
  {
    if (LayerTraits<Layer>::IsBinary)
      return -arma::dot(arma::trunc_log(arma::abs(target - input)), target);

    return -arma::dot(arma::trunc_log(input), target);
  }

}; // class CrossEntropyErrorFunction

} // namespace ann
} // namespace mlpack

#endif
