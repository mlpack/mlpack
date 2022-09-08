/**
 * @file methods/ann/regularizer/no_regularizer.hpp
 * @author Saksham Bansal
 *
 * Definition of the NoRegularizer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_NO_REGULARIZER_HPP
#define MLPACK_METHODS_ANN_NO_REGULARIZER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the NoRegularizer. This does not add any
 * regularization to the weights.
 */
class NoRegularizer
{
 public:
  /**
   * Create the regularizer object.
   */
  NoRegularizer()
  {
    // Nothing to do here.
  };

  /**
   * Calculate the gradient for regularization.
   *
   * @tparam MatType Type of weight matrix.
   * @param * (weight) The weight matrix to be regularized.
   * @param * (gradient) The calculated gradient.
   */
  template<typename MatType>
  void Evaluate(const MatType& /* weight */, MatType& /* gradient */)
  {
    // Nothing to do here.
  }

  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */)
  {
    // Nothing to do.
  }
};

} // namespace mlpack

#endif
