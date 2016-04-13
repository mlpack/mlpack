/**
 * @file qdafn_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of QDAFN class methods.
 */
#ifndef QDAFN_IMPL_HPP
#define QDAFN_IMPL_HPP

// In case it hasn't been included yet.
#include "qdafn.hpp"

#include <queue>
#include <mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp>

namespace qdafn {

// Constructor.
template<typename MatType>
QDAFN<MatType>::QDAFN(const MatType& referenceSet,
                      const size_t l,
                      const size_t m) :
    referenceSet(referenceSet),
    l(l),
    m(m)
{
  // Build tables.  This is done by drawing random points from a Gaussian
  // distribution as the vectors we project onto.  The Gaussian should have zero
  // mean and unit variance.
  mlpack::distribution::GaussianDistribution gd(referenceSet.n_rows);
  lines.set_size(referenceSet.n_rows, l);
  for (size_t i = 0; i < l; ++i)
    lines.col(i) = gd.Random();

  // Now, project each of the reference points onto each line, and collect the
  // top m elements.
  projections = referenceSet.t() * lines;

  // Loop over each projection and find the top m elements.
  sIndices.set_size(m, l);
  sValues.set_size(m, l);
  for (size_t i = 0; i < l; ++i)
  {
    arma::uvec sortedIndices = arma::sort_index(projections.col(i), "descend");

    // Grab the top m elements.
    for (size_t j = 0; j < m; ++j)
    {
      sIndices(j, i) = sortedIndices[j];
      sValues(j, i) = projections(sortedIndices[j], i);
    }
  }
}

// Search.
template<typename MatType>
void QDAFN<MatType>::Search(const MatType& querySet,
                            const size_t k,
                            arma::Mat<size_t>& neighbors,
                            arma::mat& distances)
{
  if (k > m)
    throw std::invalid_argument("QDAFN::Search(): requested k is greater than "
        "value of m!");

  neighbors.set_size(k, querySet.n_cols);
  neighbors.fill(size_t() - 1);
  distances.zeros(k, querySet.n_cols);

  // Search for each point.
  for (size_t q = 0; q < querySet.n_cols; ++q)
  {
    // Initialize a priority queue.
    // The size_t represents the index of the table, and the double represents
    // the value of l_i * S_i - l_i * query (see line 6 of Algorithm 1).
    std::priority_queue<std::pair<double, size_t>> queue;
    for (size_t i = 0; i < l; ++i)
    {
      const double val = projections(0, i) - arma::dot(querySet.col(q),
                                                       lines.col(i));
      queue.push(std::make_pair(val, i));
    }

    // To track where we are in each S table, we keep the next index to look at
    // in each table (they start at 0).
    arma::Col<size_t> tableLocations = arma::zeros<arma::Col<size_t>>(l);

    // Now that the queue is initialized, iterate over m elements.
    for (size_t i = 0; i < m; ++i)
    {
      std::pair<size_t, double> p = queue.top();
      queue.pop();

      // Get index of reference point to look at.
      size_t referenceIndex = sIndices(tableLocations[p.second], p.second);

      // Calculate distance from query point.
      const double dist = mlpack::metric::EuclideanDistance::Evaluate(
          querySet.col(q), referenceSet.col(referenceIndex));

      // Is this neighbor good enough to insert into the results?
      arma::vec queryDist = distances.unsafe_col(q);
      arma::Col<size_t> queryIndices = neighbors.unsafe_col(q);
      const size_t insertPosition =
          mlpack::neighbor::FurthestNeighborSort::SortDistance(queryDist,
          queryIndices, dist);

      // SortDistance() returns (size_t() - 1) if we shouldn't add it.
      if (insertPosition != (size_t() - 1))
        InsertNeighbor(distances, neighbors, q, insertPosition, referenceIndex,
            dist);

      // Now (line 14) get the next element and insert into the queue.  Do this
      // by adjusting the previous value.  Don't insert anything if we are at
      // the end of the search, though.
      if (i < m - 1)
      {
        tableLocations[p.second]++;
        const double val = p.first -
            projections(tableLocations[p.second] - 1, p.second) +
            projections(tableLocations[p.second], p.second);

        queue.push(std::make_pair(val, p.second));
      }
    }
  }
}

template<typename MatType>
void QDAFN<MatType>::InsertNeighbor(arma::mat& distances,
                                    arma::Mat<size_t>& neighbors,
                                    const size_t queryIndex,
                                    const size_t pos,
                                    const size_t neighbor,
                                    const double distance) const
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (distances.n_rows - 1))
  {
    const size_t len = (distances.n_rows - 1) - pos;
    memmove(distances.colptr(queryIndex) + (pos + 1),
        distances.colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(neighbors.colptr(queryIndex) + (pos + 1),
        neighbors.colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  distances(pos, queryIndex) = distance;
  neighbors(pos, queryIndex) = neighbor;
}

} // namespace qdafn

#endif
