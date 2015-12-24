/**
 * @file sse_function.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the sum squared error performance function.
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
#ifndef __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SSE_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SSE_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The sum squared error performance function measures the network's performance
 * according to the sum of squared errors.
 */
class SumSquaredErrorFunction
{
  public:
  /**
   * Computes the sum squared error function.
   *
   * @param input Input data.
   * @param target Target data.
   * @return sum of squared errors.
   */
  template<typename DataType>
  static double Error(const DataType& input, const DataType& target)
  {
    return arma::sum(arma::square(target - input));
  }

}; // class SumSquaredErrorFunction

} // namespace ann
} // namespace mlpack

#endif
