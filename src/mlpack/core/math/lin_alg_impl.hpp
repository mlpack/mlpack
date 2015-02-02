/**
 * @file lin_alg_impl.hpp
 * @author Stephen Tu
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
