/***
 * @file cosine_similarity.hpp
 * @author jeffin Sam
 *
 * The Cosine similarity metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_COSINE_SIMILARITY_HPP
#define MLPACK_CORE_METRICS_COSINE_SIMILARITY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace metric {

/**
 * Class for Cosine Similarity
 */
class CosineSimilarity
{
 public:
  /**
   * Empty constructor for CosineSimilarity Class
   */
  CosineSimilarity() { }

  /**
   * Evaluate the Cosine Similarity
   *
   * @param a First vector.
   * @param b Second vector.
   * vector could be arma::rowvec or arma::colvec
   */
  template<typename VecType>
  double Evaluate(const VecType& a, const VecType& b);
};

} // namespace metric
} // namespace mlpack

#include "cosine_similarity_impl.hpp"

#endif
