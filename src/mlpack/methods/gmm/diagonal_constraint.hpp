/**
 * @file diagonal_constraint.hpp
 * @author Ryan Curtin
 *
 * Constrain a covariance matrix to be diagonal.
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
