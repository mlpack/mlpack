/**
 * @file ra_util.hpp
 * @author Parikshit Ram
 * @author Ryan Curtin
 *
 * Utilities for rank-approximate neighbor search.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_METHODS_RANN_RA_UTIL_HPP
#define MLPACK_METHODS_RANN_RA_UTIL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

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

  /**
   * Pick up desired number of samples (with replacement) from a given range
   * of integers so that only the distinct samples are returned from
   * the range [0 - specified upper bound)
   *
   * @param numSamples Number of random samples.
   * @param rangeUpperBound The upper bound on the range of integers.
   * @param distinctSamples The list of the distinct samples.
   */
  static void ObtainDistinctSamples(const size_t numSamples,
                                    const size_t rangeUpperBound,
                                    arma::uvec& distinctSamples);
};

} // namespace neighbor
} // namespace mlpack

#endif
