/**
 * @file lin_alg_impl.hpp
 * @author Stephen Tu
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
#ifndef MLPACK_CORE_MATH_LIN_ALG_IMPL_HPP
#define MLPACK_CORE_MATH_LIN_ALG_IMPL_HPP

#include "lin_alg.hpp"

namespace mlpack {
namespace math {

inline size_t SvecIndex(size_t i, size_t j, size_t n)
{
  if (i > j)
    std::swap(i, j);
  return (j-i) + (n*(n+1) - (n-i)*(n-i+1))/2;
}

} // namespace math
} // namespace mlpack

#endif
