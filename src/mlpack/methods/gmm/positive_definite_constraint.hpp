/**
 * @file positive_definite_constraint.hpp
 * @author Ryan Curtin
 *
 * Restricts a covariance matrix to being positive definite.
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

  //! Serialize the constraint (which stores nothing, so, nothing to do).
  template<typename Archive>
  static void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace gmm
} // namespace mlpack

#endif

