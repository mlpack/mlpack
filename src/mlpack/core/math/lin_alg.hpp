/**
 * @file core/math/lin_alg.hpp
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_LIN_ALG_HPP
#define MLPACK_CORE_MATH_LIN_ALG_HPP

#include <mlpack/prereqs.hpp>
#include "ccov.hpp"
#include "random.hpp"

/**
 * Linear algebra utility functions, generally performed on matrices or vectors.
 */
namespace mlpack {

/**
 * Creates a centered matrix, where centering is done by subtracting
 * the sum over the columns (a column vector) from each column of the matrix.
 *
 * @param x Input matrix
 * @param xCentered Matrix to write centered output into
 */
inline void Center(const arma::mat& x, arma::mat& xCentered);

/**
 * Overwrites a dimension-N vector to a random vector on the unit sphere in R^N.
 */
inline void RandVector(arma::vec& v);

} // namespace mlpack

// Partially include implementation
#include "lin_alg_impl.hpp"

#endif // MLPACK_CORE_MATH_LIN_ALG_HPP
