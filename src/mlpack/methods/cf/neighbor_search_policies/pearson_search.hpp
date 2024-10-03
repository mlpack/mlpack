/**
 * @file methods/cf/neighbor_search_policies/pearson_search.hpp
 * @author Wenhao Huang
 *
 * Nearest neighbor search with pearson distance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_PEARSON_SEARCH_HPP
#define MLPACK_METHODS_CF_PEARSON_SEARCH_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

namespace mlpack {

/**
 * Nearest neighbor search with pearson distance (or furthest neighbor search
 * with pearson correlation).
 * Note that, with normalized vectors, neighbor search with pearson distance is
 * equivalent to neighbor search with Euclidean distance. Therefore, instead of
 * performing neighbor search directly with pearson distance, we first normalize
 * all vectors, and then use KNN (i.e. NeighborSearch with Euclidean distance,
 * KDTree). Pearson correlation are calculated from Euclidean distance.
 *
 * An example of how to use PearsonSearch in CF is shown below:
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
 * cf.template GetRecommendations<PearsonSearch>(10, recommendations);
 * @endcode
 */
class PearsonSearch
{
 public:
  /**
   * Constructor with reference set.  In order to use KNN(i.e. NeighborSearch
   * with Euclidean distance, KDTree), we need to normalize all vectors in
   * referenceSet. For each vector x, we first subtract mean(x) from each
   * element in x. Then, we normalize the vector to unit length.
   *
   * @param referenceSet Set of reference points.
   */
  PearsonSearch(const arma::mat& referenceSet)
  {
    // Normalize all vectors in referenceSet.
    // For each vector x, first subtract mean(x) from each element in x.
    // Then normalize the vector to unit length.
    arma::mat normalizedSet(arma::size(referenceSet));
    normalizedSet = normalise(
        referenceSet.each_row() - arma::mean(referenceSet));

    neighborSearch.Train(std::move(normalizedSet));
  }

  /**
   * Given a set of query points, find the nearest k neighbors, and return
   * similarities. Similarities are non-negative and no larger than one.
   *
   * @param query A set of query points.
   * @param k Number of neighbors to search.
   * @param neighbors Nearest neighbors.
   * @param similarities Similarities between query point and its neighbors.
   */
  void Search(const arma::mat& query, const size_t k,
              arma::Mat<size_t>& neighbors, arma::mat& similarities)
  {
    // Normalize all vectors in query.
    // For each vector x, first subtract mean(x) from each element in x.
    // Then normalize the vector to unit length.
    arma::mat normalizedQuery;
    normalizedQuery = normalise(query.each_row() - arma::mean(query));

    neighborSearch.Search(normalizedQuery, k, neighbors, similarities);

    // Resulting similarities from Search() are Euclidean distance.
    // For normalized vectors a and b, pearson(a, b) = 1 - dis(a, b) ^ 2 / 2,
    // where dis(a, b) is Euclidean distance.
    // Furthermore, we restrict the range of similarity to be [0, 1]:
    // similarities = (pearson(a,b) + 1) / 2.0. As a result we have the
    // following formula.
    similarities = 1 - pow(similarities, 2) / 4.0;
  }

 private:
  //! NeighborSearch object.
  KNN neighborSearch;
};

} // namespace mlpack

#endif
