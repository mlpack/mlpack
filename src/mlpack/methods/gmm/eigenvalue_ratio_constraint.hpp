/**
 * @file eigenvalue_ratio_constraint.hpp
 * @author Ryan Curtin
 *
 * Constrain a covariance matrix to have a certain ratio of eigenvalues.
 *
 * This file is part of MLPACK 1.0.6.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_GMM_EIGENVALUE_RATIO_CONSTRAINT_HPP
#define __MLPACK_METHODS_GMM_EIGENVALUE_RATIO_CONSTRAINT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace gmm {

/**
 * Given a vector of eigenvalue ratios, ensure that the covariance matrix always
 * has those eigenvalue ratios.  When you create this object, make sure that the
 * vector of ratios that you pass does not go out of scope, because this object
 * holds a reference to that vector instead of copying it.
 */
class EigenvalueRatioConstraint
{
 public:
  /**
   * Create the EigenvalueRatioConstraint object with the given vector of
   * eigenvalue ratios.  These ratios are with respect to the first eigenvalue,
   * which is the largest eigenvalue, so the first element of the vector should
   * be 1.  In addition, all other elements should be less than or equal to 1.
   */
  EigenvalueRatioConstraint(const arma::vec& ratios) :
      ratios(ratios)
  {
    // Check validity of ratios.
    if (std::abs(ratios[0] - 1.0) > 1e-20)
      Log::Fatal << "EigenvalueRatioConstraint::EigenvalueRatioConstraint(): "
          << "first element of ratio vector is not 1.0!" << std::endl;

    for (size_t i = 1; i < ratios.n_elem; ++i)
    {
      if (ratios[i] > 1.0)
        Log::Fatal << "EigenvalueRatioConstraint::EigenvalueRatioConstraint(): "
            << "element " << i << " of ratio vector is greater than 1.0!"
            << std::endl;
      if (ratios[i] < 0.0)
        Log::Warn << "EigenvalueRatioConstraint::EigenvalueRatioConstraint(): "
            << "element " << i << " of ratio vectors is negative and will "
            << "probably cause the covariance to be non-invertible..."
            << std::endl;
    }
  }

  /**
   * Apply the eigenvalue ratio constraint to the given covariance matrix.
   */
  void ApplyConstraint(arma::mat& covariance) const
  {
    // Eigendecompose the matrix.
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    arma::eig_sym(eigenvalues, eigenvectors, covariance);

    // Change the eigenvalues to what we are forcing them to be.  There
    // shouldn't be any negative eigenvalues anyway, so it doesn't matter if we
    // are suddenly forcing them to be positive.  If the first eigenvalue is
    // negative, well, there are going to be some problems later...
    eigenvalues = (eigenvalues[0] * ratios);

    // Reassemble the matrix.
    covariance = eigenvectors * arma::diagmat(eigenvalues) * eigenvectors.t();
  }

 private:
  //! Ratios for eigenvalues.
  const arma::vec& ratios;
};

}; // namespace gmm
}; // namespace mlpack

#endif
