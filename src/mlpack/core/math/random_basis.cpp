/**
 * @file random_basis.cpp
 * @author Ryan Curtin
 *
 * Generate a random d-dimensional basis.
 */
#include "random_basis.hpp"

using namespace arma;

namespace mlpack {
namespace math {

void RandomBasis(mat& basis, const size_t d)
{
  while(true)
  {
    // [Q, R] = qr(randn(d, d));
    // Q = Q * diag(sign(diag(R)));
    mat r;
    if (qr(basis, r, randn<mat>(d, d)))
    {
      vec rDiag(r.n_rows);
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

} // namespace math
} // namespace mlpack
