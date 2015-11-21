/**
 * @file cosine_distance_impl.hpp
 * @author Ryan Curtin
 *
 * This implements the cosine distance.
 */
#ifndef __MLPACK_CORE_KERNELS_COSINE_DISTANCE_IMPL_HPP
#define __MLPACK_CORE_KERNELS_COSINE_DISTANCE_IMPL_HPP

#include "cosine_distance.hpp"

namespace mlpack {
namespace kernel {

template<typename VecTypeA, typename VecTypeB>
double CosineDistance::Evaluate(const VecTypeA& a, const VecTypeB& b)
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

} // namespace kernel
} // namespace mlpack

#endif
