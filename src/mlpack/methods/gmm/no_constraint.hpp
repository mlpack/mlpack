/**
 * @file no_constraint.hpp
 * @author Ryan Curtin
 *
 * No constraint on the covariance matrix.
 *
 * This file is part of MLPACK 1.0.10.
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
#ifndef __MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP
#define __MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace gmm {

/**
 * This class enforces no constraint on the covariance matrix.  It's faster this
 * way, although depending on your situation you may end up with a
 * non-invertible covariance matrix.
 */
class NoConstraint
{
 public:
  //! Do nothing, and do not modify the covariance matrix.
  static void ApplyConstraint(const arma::mat& /* covariance */) { }
};

}; // namespace gmm
}; // namespace mlpack

#endif
