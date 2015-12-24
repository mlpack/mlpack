/**
 * @file random_init.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix.
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
#ifndef __MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP
#define __MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize randomly the weight matrix.
 */
class RandomInitialization
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  RandomInitialization(const double lowerBound = -1,
                       const double upperBound = 1) :
      lowerBound(lowerBound), upperBound(upperBound) { }

  /**
   * Initialize the random initialization rule with the given bound.
   * Using the negative of the bound as lower bound and the postive bound as
   * upper bound.
   *
   * @param bound The number used as lower bound
   */
  RandomInitialization(const double bound) :
      lowerBound(-std::abs(bound)), upperBound(std::abs(bound)) { }

  /**
   * Initialize randomly the elements of the specified weight matrix.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    W = lowerBound + arma::randu<arma::Mat<eT>>(rows, cols) *
        (upperBound - lowerBound);
  }

  /**
   * Initialize randomly the elements of the specified weight 3rd order tensor.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::Cube<eT>(rows, cols, slices);

    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  //! The number used as lower bound.
  const double lowerBound;

  //! The number used as upper bound.
  const double upperBound;
}; // class RandomInitialization

} // namespace ann
} // namespace mlpack

#endif
