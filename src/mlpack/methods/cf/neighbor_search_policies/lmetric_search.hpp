/**
 * @file lmetric_search.hpp
 * @author Wenhao Huang
 *
 * Nearest neighbor search with L_p distance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_LMETRIC_SEARCH_HPP
#define MLPACK_METHODS_CF_LMETRIC_SEARCH_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace cf {

/**
 * Nearest neighbor search with L_p distance.
 *
 * @tparam TPower Power of metric.
 */
template<int TPower>
class LMetricSearch
{
 public:
  using NeighborSearchType = neighbor::NeighborSearch<
      neighbor::NearestNeighborSort,
      metric::LMetric<TPower, true>>;

  /**
   * @param Set of reference points.
   */
  LMetricSearch(const arma::mat& referenceSet) : neighborSearch(referenceSet)
  { }

  /**
   * Given a set of query points, find the nearest k neighbors, and return
   * similarites. Similarities are non-negative and no larger thant one.
   *
   * @param query A set of query points.
   * @param k Number of neighbors to search.
   * @param neighbors Nearest neighbors.
   * @param similarites Similarities between query point and its neighbors.
   */
  void Search(const arma::mat& query, const size_t k,
              arma::Mat<size_t>& neighbors, arma::mat& similarities)
  {
    neighborSearch.Search(query, k, neighbors, similarities);

    // Calculate similarities from L_p distance. We restrict that similarities
    // are not larger than one.
    similarities = 1.0 / (1.0 + similarities);
  }

 private:
  //! NeighborSearch object.
  NeighborSearchType neighborSearch;
};

using EuclideanSearch = LMetricSearch<2>;

} // namespace cf
} // namespace mlpack

#endif
