/**
 * @file methods/approx_kfn/drusilla_select_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of DrusillaSelect class methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_APPROX_KFN_DRUSILLA_SELECT_IMPL_HPP
#define MLPACK_METHODS_APPROX_KFN_DRUSILLA_SELECT_IMPL_HPP

// In case it hasn't been included yet.
#include "drusilla_select.hpp"

#include <queue>
#include <mlpack/methods/neighbor_search/neighbor_search_rules.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/furthest_neighbor_sort.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <algorithm>

namespace mlpack {

// Constructor.
template<typename MatType>
DrusillaSelect<MatType>::DrusillaSelect(const MatType& referenceSet,
                                        const size_t l,
                                        const size_t m) :
    candidateSet(referenceSet.n_cols, l * m),
    candidateIndices(l * m),
    l(l),
    m(m)
{
  if (l == 0)
    throw std::invalid_argument("DrusillaSelect::DrusillaSelect(): invalid "
        "value of l; must be greater than 0!");
  else if (m == 0)
    throw std::invalid_argument("DrusillaSelect::DrusillaSelect(): invalid "
        "value of m; must be greater than 0!");

  Train(referenceSet, l, m);
}

// Constructor with no training.
template<typename MatType>
DrusillaSelect<MatType>::DrusillaSelect(const size_t l, const size_t m) :
    candidateSet(0, l * m),
    candidateIndices(l * m),
    l(l),
    m(m)
{
  if (l == 0)
    throw std::invalid_argument("DrusillaSelect::DrusillaSelect(): invalid "
        "value of l; must be greater than 0!");
  else if (m == 0)
    throw std::invalid_argument("DrusillaSelect::DrusillaSelect(): invalid "
        "value of m; must be greater than 0!");
}

// Train the model.
template<typename MatType>
void DrusillaSelect<MatType>::Train(
    const MatType& referenceSet,
    const size_t lIn,
    const size_t mIn)
{
  // Did the user specify a new size?  If so, use it.
  if (lIn > 0)
    l = lIn;
  if (mIn > 0)
    m = mIn;

  if ((l * m) > referenceSet.n_cols)
    throw std::invalid_argument("DrusillaSelect::Train(): l and m are too "
        "large!  Choose smaller values.  l*m must be smaller than the number "
        "of points in the dataset.");

  candidateSet.set_size(referenceSet.n_rows, l * m);
  candidateIndices.set_size(l * m);

  arma::vec dataMean(arma::mean(referenceSet, 1));
  arma::vec norms(referenceSet.n_cols);

  MatType refCopy(referenceSet.n_rows, referenceSet.n_cols);
  for (size_t i = 0; i < refCopy.n_cols; ++i)
  {
    refCopy.col(i) = referenceSet.col(i) - dataMean;
    norms[i] = norm(refCopy.col(i));
  }

  // Find the top m points for each of the l projections...
  for (size_t i = 0; i < l; ++i)
  {
    // Pick best index.
    arma::uword maxIndex = norms.index_max();

    arma::vec line(refCopy.col(maxIndex) / norm(refCopy.col(maxIndex)));

    // Calculate distortion and offset and make scores.
    std::vector<bool> closeAngle(referenceSet.n_cols, false);
    arma::vec sums(referenceSet.n_cols);
    for (size_t j = 0; j < referenceSet.n_cols; ++j)
    {
      if (norms[j] > 0.0)
      {
        const double offset = dot(refCopy.col(j), line);
        const double distortion = norm(refCopy.col(j) - offset * line);
        sums[j] = std::abs(offset) - std::abs(distortion);
        closeAngle[j] =
            (std::atan(distortion / std::abs(offset)) < (M_PI / 8.0));
      }
      else
      {
        sums[j] = norms[j];
      }
    }

    // Find the top m elements using a priority queue.
    using Candidate = std::pair<double, size_t>;
    struct CandidateCmp
    {
      bool operator()(const Candidate& c1, const Candidate& c2)
      {
        return c2.first < c1.first;
      }
    };

    std::vector<Candidate> clist(
        m, std::make_pair(double(-DBL_MAX), size_t(-1)));
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateCmp>
        pq(CandidateCmp(), std::move(clist));

    for (size_t j = 0; j < sums.n_elem; ++j)
    {
      Candidate c = std::make_pair(sums[j], j);
      if (CandidateCmp()(c, pq.top()))
      {
        pq.pop();
        pq.push(c);
      }
    }

    // Take the top m elements for this table.
    for (size_t j = 0; j < m; ++j)
    {
      const size_t index = pq.top().second;
      pq.pop();
      candidateSet.col(i * m + j) = referenceSet.col(index);
      candidateIndices[i * m + j] = index;

      // Mark the norm as -1 so we don't see this point again.
      norms[index] = -1.0;
    }

    // Calculate angles from the current projection.  Anything close enough,
    // mark the norm as 0.
    for (size_t j = 0; j < norms.n_elem; ++j)
      if (norms[j] > 0.0 && closeAngle[j])
        norms[j] = 0.0;
  }
}

// Search.
template<typename MatType>
void DrusillaSelect<MatType>::Search(const MatType& querySet,
                                     const size_t k,
                                     arma::Mat<size_t>& neighbors,
                                     arma::mat& distances)
{
  if (candidateSet.n_cols == 0)
    throw std::runtime_error("DrusillaSelect::Search(): candidate set not "
        "initialized!  Call Train() first.");

  if (k > (l * m))
    throw std::invalid_argument("DrusillaSelect::Search(): requested k is "
        "greater than number of points in candidate set!  Increase l or m.");

  // We'll use the NeighborSearchRules class to perform our brute-force search.
  // Note that we aren't using trees for our search, so we can use 'int' as a
  // TreeType.
  EuclideanDistance metric;
  NeighborSearchRules<FurthestNeighborSort, EuclideanDistance,
      KDTree<EuclideanDistance, EmptyStatistic, MatType>>
      rules(candidateSet, querySet, k, metric, 0, false);

  for (size_t q = 0; q < querySet.n_cols; ++q)
    for (size_t r = 0; r < candidateSet.n_cols; ++r)
      rules.BaseCase(q, r);

  rules.GetResults(neighbors, distances);

  // Map the neighbors back to their original indices in the reference set.
  for (size_t i = 0; i < neighbors.n_elem; ++i)
    neighbors[i] = candidateIndices[neighbors[i]];
}

//! Serialize the model.
template<typename MatType>
template<typename Archive>
void DrusillaSelect<MatType>::serialize(Archive& ar,
                                        const uint32_t /* version */)
{
  ar(CEREAL_NVP(candidateSet));
  ar(CEREAL_NVP(candidateIndices));
  ar(CEREAL_NVP(l));
  ar(CEREAL_NVP(m));
}

} // namespace mlpack

#endif
