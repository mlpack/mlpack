/**
 * @file core/math/random_basis.hpp
 * @author Ryan Curtin
 *
 * Generate a random d-dimensional basis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_RANDOM_BASIS_HPP
#define MLPACK_CORE_MATH_RANDOM_BASIS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Create a random d-dimensional orthogonal basis, storing it in the given
 * matrix.
 *
 * @param basis Matrix to store basis in.
 * @param d Desired number of dimensions in the basis.
 */
inline void RandomBasis(arma::mat& basis, const size_t d);

} // namespace mlpack

//! Include the implementation file.
#include "random_basis_impl.hpp"

#endif
