/**
 * @file diagonal_constraint.hpp
 * @author Ryan Curtin
 *
 * Constrain a covariance matrix to be diagonal.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP
#define __MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace gmm {

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
    arma::vec diagonal = covariance.diag();
    covariance = arma::diagmat(diagonal);
  }
};

}; // namespace gmm
}; // namespace mlpack

#endif
