/**
 * @file qdafn_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of QDAFN class methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_APPROX_KFN_QDAFN_IMPL_HPP
#define MLPACK_METHODS_APPROX_KFN_QDAFN_IMPL_HPP

// In case it hasn't been included yet.
#include "qdafn.hpp"

#include <queue>
#include <mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp>

namespace mlpack {
namespace neighbor {

// Non-training constructor.
template<typename MatType>
QDAFN<MatType>::QDAFN(const size_t l, const size_t m) : l(l), m(m)
{
  if (l == 0)
    throw std::invalid_argument("QDAFN::QDAFN(): l must be greater than 0!");
  if (m == 0)
    throw std::invalid_argument("QDAFN::QDAFN(): m must be greater than 0!");
}

// Constructor.
template<typename MatType>
QDAFN<MatType>::QDAFN(const MatType& referenceSet,
                      const size_t l,
                      const size_t m) :
    l(l),
    m(m)
{
  if (l == 0)
    throw std::invalid_argument("QDAFN::QDAFN(): l must be greater than 0!");
  if (m == 0)
    throw std::invalid_argument("QDAFN::QDAFN(): m must be greater than 0!");

  Train(referenceSet);
}

// Train the object.
template<typename MatType>
void QDAFN<MatType>::Train(const MatType& referenceSet,
                           const size_t lIn,
                           const size_t mIn)
{
  if (lIn != 0)
    l = lIn;
  if (mIn != 0)
    m = mIn;

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
  candidateSet.resize(l);
  for (size_t i = 0; i < l; ++i)
  {
    candidateSet[i].set_size(referenceSet.n_rows, m);
    arma::uvec sortedIndices = arma::sort_index(projections.col(i), "descend");

    // Grab the top m elements.
    for (size_t j = 0; j < m; ++j)
    {
      sIndices(j, i) = sortedIndices[j];
      sValues(j, i) = projections(sortedIndices[j], i);
      candidateSet[i].col(j) = referenceSet.col(sortedIndices[j]);
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
      const double val = sValues(0, i) - arma::dot(querySet.col(q),
          lines.col(i));
      queue.push(std::make_pair(val, i));
    }

    // To track where we are in each S table, we keep the next index to look at
    // in each table (they start at 0).
    arma::Col<size_t> tableLocations = arma::zeros<arma::Col<size_t>>(l);

    // Now that the queue is initialized, iterate over m elements.
    std::vector<std::pair<double, size_t>> v(k, std::make_pair(-1.0,
        size_t(-1)));
    std::priority_queue<std::pair<double, size_t>>
        resultsQueue(std::less<std::pair<double, size_t>>(), std::move(v));
    for (size_t i = 0; i < m; ++i)
    {
      std::pair<size_t, double> p = queue.top();
      queue.pop();

      // Get index of reference point to look at.
      const size_t tableIndex = tableLocations[p.second];

      // Calculate distance from query point.
      const double dist = mlpack::metric::EuclideanDistance::Evaluate(
          querySet.col(q), candidateSet[p.second].col(tableIndex));

      // Is this neighbor good enough to insert into the results?
      if (dist > resultsQueue.top().first)
      {
        resultsQueue.pop();
        resultsQueue.push(std::make_pair(dist, sIndices(tableIndex, p.second)));
      }

      // Now (line 14) get the next element and insert into the queue.  Do this
      // by adjusting the previous value.  Don't insert anything if we are at
      // the end of the search, though.
      if (i < m - 1)
      {
        tableLocations[p.second]++;
        const double val = p.first - sValues(tableIndex, p.second) +
            sValues(tableIndex + 1, p.second);

        queue.push(std::make_pair(val, p.second));
      }
    }

    // Extract the results.
    for (size_t j = 1; j <= k; ++j)
    {
      neighbors(k - j, q) = resultsQueue.top().second;
      distances(k - j, q) = resultsQueue.top().first;
      resultsQueue.pop();
    }
  }
}

template<typename MatType>
template<typename Archive>
void QDAFN<MatType>::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(l, "l");
  ar & CreateNVP(m, "m");
  ar & CreateNVP(lines, "lines");
  ar & CreateNVP(projections, "projections");
  ar & CreateNVP(sIndices, "sIndices");
  ar & CreateNVP(sValues, "sValues");
  if (Archive::is_loading::value)
    candidateSet.clear();
  ar & CreateNVP(candidateSet, "candidateSet");
}

} // namespace neighbor
} // namespace mlpack

#endif
