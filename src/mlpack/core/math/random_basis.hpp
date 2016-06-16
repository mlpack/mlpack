/**
 * @file random_basis.hpp
 * @author Ryan Curtin
 *
 * Generate a random d-dimensional basis.
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
#ifndef MLPACK_CORE_MATH_RANDOM_BASIS_HPP
#define MLPACK_CORE_MATH_RANDOM_BASIS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace math {

/**
 * Create a random d-dimensional orthogonal basis, storing it in the given
 * matrix.
 *
 * @param basis Matrix to store basis in.
 * @param d Desired number of dimensions in the basis.
 */
void RandomBasis(arma::mat& basis, const size_t d);

} // namespace math
} // namespace mlpack

#endif
