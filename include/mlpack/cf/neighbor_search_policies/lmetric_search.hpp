/**
 * @file methods/cf/neighbor_search_policies/lmetric_search.hpp
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
#include <mlpack/core/distances/lmetric.hpp>

namespace mlpack {

/**
 * Nearest neighbor search with L_p distance.
 *
 * An example of how to use LMetricSearch in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * CFType<> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.template GetRecommendations<LMetricSearch<2>>(10, recommendations);
 * @endcode
 *
 * @tparam TPower Power of metric.
 */
template<int TPower>
class LMetricSearch
{
 public:
  using NeighborSearchType = NeighborSearch<
      NearestNeighborSort, LMetric<TPower, true>>;

  /**
   * @param referenceSet Set of reference points.
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
   * @param similarities Similarities between query point and its neighbors.
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

} // namespace mlpack

#endif
