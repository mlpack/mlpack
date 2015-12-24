/**
 * @file mean_pooling.hpp
 * @author Shangtong Zhang
 *
 * Definition of the MeanPooling class, which implements mean pooling.
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
#ifndef __MLPACK_METHODS_ANN_POOLING_RULES_MEAN_POOLING_HPP
#define __MLPACK_METHODS_ANN_POOLING_RULES_MEAN_POOLING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/*
 * The mean pooling rule for convolution neural networks. Average all values
 * within the receptive block.
 */
class MeanPooling
{
 public:
  /*
   * Return the average value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   */
  template<typename MatType>
  double Pooling(const MatType& input)
  {
    return arma::mean(arma::mean(input));
  }

  /*
   * Set the average value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   * @param value The unpooled value.
   * @param output The unpooled output data.
   */
  template<typename MatType>
  void Unpooling(const MatType& input, const double value, MatType& output)
  {
    output = MatType(input.n_rows, input.n_cols);
    output.fill(value / input.n_elem);
  }
};

} // namespace ann
} // namespace mlpack

#endif
