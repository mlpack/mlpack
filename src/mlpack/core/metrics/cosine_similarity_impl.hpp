/***
 * @file cosine_similarity_impl.hpp
 * @author jeffin Sam
 *
 * The Cosine similarity metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_COSINE_SIMILARITY_IMPL_HPP
#define MLPACK_CORE_METRICS_COSINE_SIMILARITY_IMPL_HPP

#include "cosine_similarity.hpp"

namespace mlpack {
namespace metric {

template<typename VecType>
double CosineSimilarity::Evaluate(const VecType& a, const VecType& b)
{
  return arma::accu(arma::normalise(a) % arma::normalise(b));
}

} // namespace metric
} // namespace mlpack

#endif
