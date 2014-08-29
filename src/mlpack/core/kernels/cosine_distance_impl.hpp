/**
 * @file cosine_distance_impl.hpp
 * @author Ryan Curtin
 *
 * This implements the cosine distance.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_KERNELS_COSINE_DISTANCE_IMPL_HPP
#define __MLPACK_CORE_KERNELS_COSINE_DISTANCE_IMPL_HPP

#include "cosine_distance.hpp"

namespace mlpack {
namespace kernel {

template<typename VecType>
double CosineDistance::Evaluate(const VecType& a, const VecType& b)
{
  // Since we are using the L2 inner product, this is easy.  But we have to make
  // sure we aren't dividing by zero (if we are, then the cosine similarity is
  // 0: we reason this value because the cosine distance is just a normalized
  // dot product; take away the normalization, and if ||a|| or ||b|| is equal to
  // 0, then a^T b is zero too).
  const double denominator = norm(a, 2) * norm(b, 2);
  if (denominator == 0.0)
    return 0;
  else
    return dot(a, b) / denominator;
}

}; // namespace kernel
}; // namespace mlpack

#endif
