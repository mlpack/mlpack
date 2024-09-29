/**
 * @file methods/ann/regularizer/orthogonal_regularizer.hpp
 * @author Saksham Bansal
 *
 * Definition of the OrthogonalRegularizer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_HPP
#define MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the OrthogonalRegularizer. Orthogonality of weights is a
 * desirable property because multiplication by an orthogonal matrix leaves
 * the norm of the matrix unchanged. The orthogonal regularization technique
 * encourages weights to be orthogonal.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{WanICML2013,
 *   title={Neural Photo Editing with Introspective Adversarial Networks},
 *   booktitle = {5th International Conference on Learning Representations
 *                (ICLR - 17)},
 *   author = {Andrew Brock and Theodore Lim and J.M. Ritchie and Nick Weston},
 *   year = {2017}
 * }
 * @endcode
 */
class OrthogonalRegularizer
{
 public:
  /**
   * Create the regularizer object.
   *
   * @param factor The factor for regularization.
   */
  OrthogonalRegularizer(double factor = 1.0);

  /**
   * Calculate the gradient for regularization.
   *
   * @tparam MatType Type of weight matrix.
   * @param weight The weight matrix to be regularized.
   * @param gradient The calculated gradient.
   */
  template<typename MatType>
  void Evaluate(const MatType& weight, MatType& gradient);

  //! Serialize the regularizer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! The constant for the regularization
  double factor;
};

} // namespace mlpack

// Include implementation.
#include "orthogonal_regularizer_impl.hpp"

#endif
