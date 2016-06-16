/**
 * @file random_basis.cpp
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
