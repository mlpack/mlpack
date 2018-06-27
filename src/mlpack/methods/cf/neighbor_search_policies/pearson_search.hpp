/**
 * @file pearson_search.hpp
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
namespace cf {

/**
 * Nearest neighbor search with pearson distance (or furthest neighbor search
 * with pearson correlation).
 * Note that, with normalized vectors, neighbor search with pearson distance is
 * equivalent to neighbor search with Euclidean distance. Therefore, instead
 * of performing neighbor search directly with pearson distance, we first
 * normalize all vectors, and then use neighbor::KNN (i.e. NeighborSearch with
 * Euclidean distance, KDTree). Pearson correlation are calculated from
 * Euclidean distance.
 */
class PearsonSearch
{
 public:
  /**
   * Constructor with reference set.
   * In order to use neighbor::KNN(i.e. NeighborSearch with Euclidean distance,
   * KDTree), we need to normalize all vectors in referenceSet. For each vector
   * x, we first subtract mean(x) from each element in x. Then, we normalize
   * the vector to unit length.
   *
   * @param Set of reference points.
   */
  PearsonSearch(const arma::mat& referenceSet)
  {
    // Normalize all vectors in referenceSet.
    // For each vector x, first subtract mean(x) from each element in x.
    // Then normalize the vector to unit length.
    arma::mat normalizedSet(arma::size(referenceSet));
    for (size_t i = 0; i < referenceSet.n_cols; i++)
    {
      // Subtract mean(x) from each element in x.
      normalizedSet.col(i) =
          referenceSet.col(i) - arma::mean(referenceSet.col(i));
      // Normalize the vector to unit length.
      normalizedSet.col(i) = arma::normalise(normalizedSet.col(i), 2);
    }

    neighborSearch.Train(std::move(normalizedSet));
  }

  /**
   * Given a set of query points, find the nearest k neighbors, and return
   * similarities. Similarities are non-negative and no larger than one.
   *
   * @param query A set of query points.
   * @param k Number of neighbors to search.
   * @param neighbors Nearest neighbors.
   * @param similarites Similarities between query point and its neighbors.
   */
  void Search(const arma::mat& query, const size_t k,
              arma::Mat<size_t>& neighbors, arma::mat& similarities)
  {
    // Normalize all vectors in query.
    // For each vector x, first subtract mean(x) from each element in x.
    // Then normalize the vector to unit length.
    arma::mat normalizedQuery(arma::size(query));
    for (size_t i = 0; i < query.n_cols; i++)
    {
      // Subtract mean(x) from each element in x.
      normalizedQuery.col(i) = query.col(i) - arma::mean(query.col(i));
      // Normalize the vector to unit length.
      normalizedQuery.col(i) = arma::normalise(normalizedQuery.col(i), 2);
    }

    neighborSearch.Search(normalizedQuery, k, neighbors, similarities);

    // Resulting similarities from Search() are Euclidean distance.
    // For normalized vectors a and b, pearson(a, b) = 1 - dis(a, b) ^ 2 / 2,
    // where dis(a, b) is Euclidean distance.
    similarities = 1 - arma::pow(similarities, 2) / 2.0;

    // The range of pearson correlation is [-1, 1]. We restrict the range of
    // similarity to be [0, 1].
    similarities = (similarities + 1) / 2.0;
  }

 private:
  //! NeighborSearch object.
  neighbor::KNN neighborSearch;
};

} // namespace cf
} // namespace mlpack

#endif
