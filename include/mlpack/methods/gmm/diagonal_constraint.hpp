/**
 * @file methods/gmm/diagonal_constraint.hpp
 * @author Ryan Curtin
 *
 * Constrain a covariance matrix to be diagonal.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP
#define MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Force a covariance matrix to be diagonal.
 */
class DiagonalConstraint
{
 public:
  //! Force a covariance matrix to be diagonal.
  static void ApplyConstraint(arma::mat& covariance)
  {
    // Save the diagonal only.
    covariance = arma::diagmat(arma::clamp(covariance.diag(), 1e-10, DBL_MAX));
  }

  /**
   * Apply the diagonal constraint to the given diagonal covariance matrix
   * (which is represented as a vector), and ensure each value on the diagonal
   * is at least 1e-10.
   */
  static void ApplyConstraint(arma::vec& diagCovariance)
  {
    // Although the covariance is already diagonal, clamp it to ensure each
    // value is at least 1e-10.
    diagCovariance = arma::clamp(diagCovariance, 1e-10, DBL_MAX);
  }

  //! Serialize the constraint (which holds nothing, so, nothing to do).
  template<typename Archive>
  static void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
