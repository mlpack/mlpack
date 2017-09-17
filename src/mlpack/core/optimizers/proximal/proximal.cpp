/**
 * @file proximal.cpp
 * @author Chenzhe Diao
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "proximal.hpp"

namespace mlpack {
namespace optimization {

/**
 * Projection of the vector v onto l1 ball with norm tau.
 * See the paper:
 * @code
 * @inproceedings{DucShaSin2008Efficient,
 *    author       = {Duchi, John and Shalev-Shwartz, Shai and Singer,
 *                    Yoram and Chandra, Tushar},
 *    booktitle    = {Proceedings of the 25th international conference on
 *                    Machine learning},
 *    organization = {ACM},
 *    pages        = {272--279},
 *    title        = {Efficient projections onto the l 1-ball for learning in
 *                    high dimensions},
 *    year         = {2008}}
 * @endcode
 *
 * This is just a soft thresholding.
 */
void Proximal::ProjectToL1Ball(arma::vec& v, double tau)
{
  arma::vec simplexSol = arma::abs(v);

  // Already with L1 norm <= tau.
  if (arma::accu(simplexSol) <= tau)
    return;

  simplexSol = arma::sort(simplexSol, "descend");
  arma::vec simplexSum = arma::cumsum(simplexSol);

  double nu = 0;
  size_t rho;
  for (size_t j = 1; j <= simplexSol.n_rows; j++)
  {
    rho = simplexSol.n_rows - j;
    nu = simplexSol(rho) - (simplexSum(rho) - tau)/(rho + 1);
    if (nu > 0)
      break;
  }
  double theta = (simplexSum(rho) - tau)/rho;

  // Threshold on absolute value of v with theta.
  for (arma::uword j = 0; j< simplexSol.n_rows; j++)
  {
    if (v(j) >= 0.0)
      v(j) = std::max(v(j) - theta, 0.0);
    else
      v(j) = std::min(v(j) + theta, 0.0);
  }
}

/**
 * Approximate the vector v with a tau-sparse vector.
 * This is a hard-thresholding.
 */
void Proximal::ProjectToL0Ball(arma::vec& v, int tau)
{
  arma::uvec indices = arma::sort_index(arma::abs(v));
  arma::uword numberToKill = v.n_elem - tau;

  for (arma::uword i = 0; i < numberToKill; i++)
    v(indices(i)) = 0.0;
}

} // namespace optimization
} // namespace mlpack
