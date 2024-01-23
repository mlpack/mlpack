/**
 * @file core/math/lin_alg_impl.hpp
 * @author Stephen Tu
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_LIN_ALG_IMPL_HPP
#define MLPACK_CORE_MATH_LIN_ALG_IMPL_HPP

#include "lin_alg.hpp"

namespace mlpack {

/**
 * Creates a centered matrix, where centering is done by subtracting
 * the sum over the columns (a column vector) from each column of the matrix.
 *
 * @param x Input matrix
 * @param xCentered Matrix to write centered output into
 */
inline void Center(const arma::mat& x, arma::mat& xCentered)
{
  // Get the mean of the elements in each row.
  arma::vec rowMean = arma::sum(x, 1) / x.n_cols;

  xCentered = x - arma::repmat(rowMean, 1, x.n_cols);
}

/**
 * Overwrites a dimension-N vector to a random vector on the unit sphere in R^N.
 */
inline void RandVector(arma::vec& v)
{
  for (size_t i = 0; i + 1 < v.n_elem; i += 2)
  {
    double a = Random();
    double b = Random();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    v[i]     = first_term * cos(second_term);
    v[i + 1] = first_term * sin(second_term);
  }

  if ((v.n_elem % 2) == 1)
  {
    v[v.n_elem - 1] = sqrt(-2 * log(Random())) * cos(2 * M_PI * Random());
  }

  v /= sqrt(dot(v, v));
}

} // namespace mlpack

#endif
