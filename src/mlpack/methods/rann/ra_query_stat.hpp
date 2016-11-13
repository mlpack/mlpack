/**
 * @file ra_query_stat.hpp
 * @author Parikshit Ram
 *
 * Defines the RAQueryStat class, which is the statistic used for
 * rank-approximate nearest neighbor search (RASearch).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_QUERY_STAT_HPP
#define MLPACK_METHODS_RANN_RA_QUERY_STAT_HPP

#include <mlpack/core.hpp>

#include <mlpack/core/tree/binary_space_tree.hpp>

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>

namespace mlpack {
namespace neighbor {

/**
 * Extra data for each node in the tree.  For neighbor searches, each node only
 * needs to store a bound on neighbor distances.
 *
 * Every query is required to make a minimum number of samples to guarantee the
 * desired approximation error. The 'numSamplesMade' keeps track of the minimum
 * number of samples made by all queries in the node in question.
 */
template<typename SortPolicy>
class RAQueryStat
{
 public:
  /**
   * Initialize the statistic with the worst possible distance according to our
   * sorting policy.
   */
  RAQueryStat() : bound(SortPolicy::WorstDistance()), numSamplesMade(0) { }

  /**
   * Initialization for a node.
   */
  template<typename TreeType>
  RAQueryStat(const TreeType& /* node */) :
    bound(SortPolicy::WorstDistance()),
    numSamplesMade(0)
  { }

  //! Get the bound.
  double Bound() const { return bound; }
  //! Modify the bound.
  double& Bound() { return bound; }

  //! Get the number of samples made.
  size_t NumSamplesMade() const { return numSamplesMade; }
  //! Modify the number of samples made.
  size_t& NumSamplesMade() { return numSamplesMade; }

  //! Serialize the statistic.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(bound, "bound");
    ar & data::CreateNVP(numSamplesMade, "numSamplesMade");
  }

 private:
  //! The bound on the node's neighbor distances.
  double bound;
  //! The minimum number of samples made by any query in this node.
  size_t numSamplesMade;
};

} // namespace neighbor
} // namespace mlpack

#endif
