/**
 * @file orthogonal_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the orthogonal matrix initialization method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP

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
