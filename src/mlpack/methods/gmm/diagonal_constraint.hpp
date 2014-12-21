/**
 * @file diagonal_constraint.hpp
 * @author Ryan Curtin
 *
 * Constrain a covariance matrix to be diagonal.
 *
 * This file is part of MLPACK 1.0.9.
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
