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

template<typename VecType>
double CosineDistance::Evaluate(const VecType& a, const VecType& b)
{
  // Since we are using the L2 inner product, this is easy.
  return 1 - dot(a, b) / (norm(a, 2) * norm(b, 2));
}

}; // namespace kernel
}; // namespace mlpack

#endif
