/**
 * @file orthogonal_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the orthogonal matrix initialization method.
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
#ifndef __MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP
#define __MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize the weight matrix with the orthogonal
 * matrix initialization
 */
class OrthogonalInitialization
{
 public:
  /**
   * Initialize the orthogonal matrix initialization rule with the given gain.
   *
   * @param gain The gain value.
   */
  OrthogonalInitialization(const double gain = 1.0) : gain(gain) { }

  /**
   * Initialize the elements of the specified weight matrix with the orthogonal
   * matrix initialization method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    arma::Mat<eT> V;
    arma::Col<eT> s;

    arma::svd_econ(W, s, V, arma::randu<arma::Mat<eT> >(rows, cols));
    W *= gain;
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * orthogonal matrix initialization method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slices Number of slices.
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
  //! The number used as gain.
  const double gain;
}; // class OrthogonalInitialization


} // namespace ann
} // namespace mlpack

#endif
