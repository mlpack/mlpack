/**
 * @file random_basis.hpp
 * @author Ryan Curtin
 *
 * Generate a random d-dimensional basis.
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
MLPACK_API
void RandomBasis(arma::mat& basis, const size_t d);

} // namespace math
} // namespace mlpack

#endif
