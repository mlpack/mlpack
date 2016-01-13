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
 * Given a covariance matrix, force the matrix to be positive definite.  Also
 * force a minimum value on the diagonal, so that even if the matrix is
 * invertible, it doesn't 
 */
class PositiveDefiniteConstraint
{
 public:
  /**
   * Apply the positive definiteness constraint to the given covariance matrix,
   * and ensure each value on the diagonal is at least 1e-50.
   *
   * @param covariance Covariance matrix.
   */
  static void ApplyConstraint(arma::mat& covariance)
  {
    // Make sure each diagonal element is at least 1e-50.
    for (size_t i = 0; i < covariance.n_cols; ++i)
      if (std::abs(covariance(i, i)) < 1e-50)
        covariance(i, i) = 1e-50;

    // Realistically, all we care about is that we can perform a Cholesky
    // decomposition of the matrix, so that FactorCovariance() doesn't fail
    // later.  Therefore, that's what we'll do to check for positive
    // definiteness...
    //
    // Note that other techniques like checking the determinant *could* work,
    // but floating-point errors mean that various decompositions may start to
    // fail when the matrix gets close to being indefinite.  This is why we test
    // with chol() and not something else, since that's what will be used later.
    //
    // We also need to make sure that the errors go to nowhere, so we have to
    // call set_stream_err2()...
    std::ostringstream oss;
    std::ostream& originalStream = arma::get_stream_err2();
    arma::set_stream_err2(oss); // Thus, errors won't be displayed.

    arma::mat covLower;
    #if (ARMA_VERSION_MAJOR < 4) || \
        ((ARMA_VERSION_MAJOR == 4) && (ARMA_VERSION_MINOR < 500))
    if (!arma::chol(covLower, covariance))
    #else
    if (!arma::chol(covLower, covariance, "lower"))
    #endif
    {
      Log::Debug << "Covariance matrix is not positive definite.  Adding "
          << "perturbation." << std::endl;

      double perturbation = 1e-15;
      #if (ARMA_VERSION_MAJOR < 4) || \
          ((ARMA_VERSION_MAJOR == 4) && (ARMA_VERSION_MAJOR < 500))
      while (!arma::chol(covLower, covariance))
      #else
      while (!arma::chol(covLower, covariance, "lower"))
      #endif
      {
        covariance.diag() += perturbation;
        perturbation *= 10;
      }
    }

    // Restore the original stream state.
    arma::set_stream_err2(originalStream);
  }

  //! Serialize the constraint (which stores nothing, so, nothing to do).
  template<typename Archive>
  static void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace gmm
} // namespace mlpack

#endif

