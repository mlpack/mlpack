/**
 * @file cosine_distance.hpp
 * @author Ryan Curtin
 *
 * This implements the cosine distance (or cosine similarity) between two
 * vectors, which is a measure of the angle between the two vectors.
 */
#ifndef __MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP
#define __MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The cosine distance (or cosine similarity).  It is defined by
 *
 * @f[
 * d(a, b) = 1 - \frac{a^T b}{|| a || || b ||}
 * @f]
 *
 * and this class assumes the standard L2 inner product.  In the future it may
 * support more.
 */
class CosineDistance
{
 public:
  /**
   * Default constructor does nothing, but is required to satisfy the Kernel
   * policy.
   */
  CosineDistance() { }

  /**
   * Computes the cosine distance between two points.
   *
   * @param a First vector.
   * @param b Second vector.
   * @return d(a, b).
   */
  template<typename VecType>
  static double Evaluate(const VecType& a, const VecType& b);
};

}; // namespace kernel
}; // namespace mlpack

// Include implementation.
#include "cosine_distance_impl.hpp"

#endif
