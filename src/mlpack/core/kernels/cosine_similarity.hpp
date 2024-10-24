/**
 * @file core/kernels/cosine_similarity.hpp
 * @author Ryan Curtin
 *
 * This implements the cosine similarity between two vectors, which is a measure
 * of the angle between the two vectors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_COSINE_SIMILARITY_HPP
#define MLPACK_CORE_KERNELS_COSINE_SIMILARITY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>

namespace mlpack {

/**
 * The cosine distance (or cosine similarity).  It is defined by
 *
 * @f[
 * d(a, b) = \frac{a^T b}{|| a || || b ||}
 * @f]
 *
 * and this class assumes the standard L2 inner product.
 */
class CosineSimilarity
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
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

//! Kernel traits for the cosine distance.
template<>
class KernelTraits<CosineSimilarity>
{
 public:
  //! The cosine kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;

  //! The cosine kernel doesn't include a squared distance.
  static const bool UsesSquaredDistance = false;
};

// This name is deprecated and can be removed in mlpack 5.0.0.
using CosineDistance = CosineSimilarity;

} // namespace mlpack

// Include implementation.
#include "cosine_similarity_impl.hpp"

#endif
