/**
 * @file methods/gmm/no_constraint.hpp
 * @author Ryan Curtin
 *
 * No constraint on the covariance matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP
#define MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class enforces no constraint on the covariance matrix.  It's faster this
 * way, although depending on your situation you may end up with a
 * non-invertible covariance matrix.
 */
class NoConstraint
{
 public:
  //! Do nothing, and do not modify the covariance matrix.
  static void ApplyConstraint(const arma::mat& /* covariance */) { }

  //! Serialize the object (nothing to do).
  template<typename Archive>
  static void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
