/**
 * @file core/math/random_basis_impl.hpp
 * @author Ryan Curtin
 *
 * Generate a random d-dimensional basis.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "random_basis.hpp"

namespace mlpack {

inline void RandomBasis(arma::mat& basis, const size_t d)
{
  while (true)
  {
    // [Q, R] = qr(randn(d, d));
    // Q = Q * diag(sign(diag(R)));
    arma::mat r;
    if (qr(basis, r, arma::randn<arma::mat>(d, d)))
    {
      arma::vec rDiag(r.n_rows);
      for (size_t i = 0; i < rDiag.n_elem; ++i)
      {
        if (r(i, i) < 0)
          rDiag(i) = -1;
        else if (r(i, i) > 0)
          rDiag(i) = 1;
        else
          rDiag(i) = 0;
      }

      basis *= diagmat(rDiag);

      // Check if the determinant is positive.
      if (det(basis) >= 0)
        break;
    }
  }
}

} // namespace mlpack
