/**
 * @file positive_definite_constraint.hpp
 * @author Ryan Curtin
 *
 * Restricts a covariance matrix to being positive definite.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP
#define MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace gmm {

/**
 * Given a covariance matrix, force the matrix to be positive definite.  Also
 * force a minimum value on the diagonal, so that even if the matrix is
 * invertible, it doesn't cause problems with Cholesky decompositions.  The
 * forcing here is also done in order to bring the condition number of the
 * matrix under 1e5 (10k), which should help with numerical stability.
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
    // What we want to do is make sure that the matrix is positive definite and
    // that the condition number isn't too large.  We also need to ensure that
    // the covariance matrix is not too close to zero (hence, we ensure that all
    // eigenvalues are at least 1e-50).
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, covariance);

    // If the matrix is not positive definite or if the condition number is
    // large, we must project it back onto the cone of positive definite
    // matrices with reasonable condition number (I'm picking 1e5 here, not for
    // any particular reason).
    if ((eigval[0] < 0.0) || ((eigval[eigval.n_elem - 1] / eigval[0]) > 1e5) ||
        (eigval[eigval.n_elem - 1] < 1e-50))
    {
      // Project any negative eigenvalues back to non-negative, and project any
      // too-small eigenvalues to a large enough value.  Make them as small as
      // possible to satisfy our constraint on the condition number.
      const double minEigval = std::max(eigval[eigval.n_elem - 1] / 1e5, 1e-50);
      for (size_t i = 0; i < eigval.n_elem; ++i)
        eigval[i] = std::max(eigval[i], minEigval);

      // Now reassemble the covariance matrix.
      covariance = eigvec * arma::diagmat(eigval) * eigvec.t();
    }
  }

  //! Serialize the constraint (which stores nothing, so, nothing to do).
  template<typename Archive>
  static void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace gmm
} // namespace mlpack

#endif

