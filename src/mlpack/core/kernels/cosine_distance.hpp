/**
 * @file cosine_distance.hpp
 * @author Ryan Curtin
 *
 * This implements the cosine distance (or cosine similarity) between two
 * vectors, which is a measure of the angle between the two vectors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP
#define MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The cosine distance (or cosine similarity).  It is defined by
 *
 * @f[
 * d(a, b) = \frac{a^T b}{|| a || || b ||}
 * @f]
 *
 * and this class assumes the standard L2 inner product.
 */
class CosineDistance
{
 public:
  /**
   * Computes the cosine distance between two points.
   *
   * @param a First vector.
   * @param b Second vector.
   * @return d(a, b).
   */
  template<typename VecTypeA, typename VecTypeB>
  static double Evaluate(const VecTypeA& a, const VecTypeB& b);

  //! Serialize the class (there's nothing to save).
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */) { }
};

//! Kernel traits for the cosine distance.
template<>
class KernelTraits<CosineDistance>
{
 public:
  //! The cosine kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;

  //! The cosine kernel doesn't include a squared distance.
  static const bool UsesSquaredDistance = false;
};

} // namespace kernel
} // namespace mlpack

// Include implementation.
#include "cosine_distance_impl.hpp"

#endif
