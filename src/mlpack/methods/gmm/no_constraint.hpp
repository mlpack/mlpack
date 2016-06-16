/**
 * @file no_constraint.hpp
 * @author Ryan Curtin
 *
 * No constraint on the covariance matrix.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP
#define MLPACK_METHODS_GMM_NO_CONSTRAINT_HPP

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

  //! Serialize the object (nothing to do).
  template<typename Archive>
  static void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

} // namespace gmm
} // namespace mlpack

#endif
