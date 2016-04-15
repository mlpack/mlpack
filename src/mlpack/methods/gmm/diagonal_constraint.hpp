/**
 * @file diagonal_constraint.hpp
 * @author Ryan Curtin
 *
 * Constrain a covariance matrix to be diagonal.
 */
#ifndef MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP
#define MLPACK_METHODS_GMM_DIAGONAL_CONSTRAINT_HPP

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

  //! Serialize the constraint (which holds nothing, so, nothing to do).
  template<typename Archive>
  static void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace gmm
} // namespace mlpack

#endif
