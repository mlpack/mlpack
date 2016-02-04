/**
 * @file lin_alg_impl.hpp
 * @author Stephen Tu
 *
 * This file is part of mlpack 2.0.1.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_MATH_LIN_ALG_IMPL_HPP
#define __MLPACK_CORE_MATH_LIN_ALG_IMPL_HPP

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
