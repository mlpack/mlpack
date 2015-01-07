/**
 * @file positive_definite_constraint.hpp
 * @author Ryan Curtin
 *
 * Restricts a covariance matrix to being positive definite.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP
#define __MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace gmm {

/**
 * Given a covariance matrix, force the matrix to be positive definite.
 */
class PositiveDefiniteConstraint
{
 public:
  /**
   * Apply the positive definiteness constraint to the given covariance matrix.
   *
   * @param covariance Covariance matrix.
   */
  static void ApplyConstraint(arma::mat& covariance)
  {
    // TODO: make this more efficient.
    if (arma::det(covariance) <= 1e-50)
    {
      Log::Debug << "Covariance matrix is not positive definite.  Adding "
          << "perturbation." << std::endl;

      double perturbation = 1e-30;
      while (arma::det(covariance) <= 1e-50)
      {
        covariance.diag() += perturbation;
        perturbation *= 10;
      }
    }
  }
};

}; // namespace gmm
}; // namespace mlpack

#endif

