/**
 * @file no_constraint.hpp
 * @author Ryan Curtin
 *
 * No constraint on the covariance matrix.
 */
#ifndef __MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP
#define __MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace gmm {

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
  static void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace gmm
} // namespace mlpack

#endif
