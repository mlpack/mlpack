/**
 * @file methods/rann/ra_util.hpp
 * @author Parikshit Ram
 * @author Ryan Curtin
 *
 * Utilities for rank-approximate neighbor search.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_UTIL_HPP
#define MLPACK_METHODS_RANN_RA_UTIL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

class RAUtil
{
 public:
  /**
   * Compute the minimum number of samples required to guarantee
   * the given rank-approximation and success probability.
   *
   * @param n Size of the set to be sampled from.
   * @param k The number of neighbors required within the rank-approximation.
   * @param tau The rank-approximation in percentile of the data.
   * @param alpha The success probability desired.
   */
  static size_t MinimumSamplesReqd(const size_t n,
                                   const size_t k,
                                   const double tau,
                                   const double alpha);

  /**
   * Compute the success probability of obtaining 'k'-neighbors from a
   * set of size 'n' within the top 't' neighbors if 'm' samples are made.
   *
   * @param n Size of the set being sampled from.
   * @param k The number of neighbors required within the rank-approximation.
   * @param m The number of random samples.
   * @param t The desired rank-approximation.
   */
  static double SuccessProbability(const size_t n,
                                   const size_t k,
                                   const size_t m,
                                   const size_t t);
};

} // namespace mlpack

// Include implementation.
#include "ra_util_impl.hpp"

#endif
