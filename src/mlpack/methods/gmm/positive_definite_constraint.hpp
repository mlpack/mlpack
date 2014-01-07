/**
 * @file positive_definite_constraint.hpp
 * @author Ryan Curtin
 *
 * Restricts a covariance matrix to being positive definite.
 *
 * This file is part of MLPACK 1.0.8.
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
#ifndef __MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP
#define __MLPACK_METHODS_GMM_POSITIVE_DEFINITE_CONSTRAINT_HPP

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
    if (det(covariance) <= 1e-50)
    {
      Log::Debug << "Covariance matrix is not positive definite.  Adding "
          << "perturbation." << std::endl;

      double perturbation = 1e-30;
      while (det(covariance) <= 1e-50)
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
