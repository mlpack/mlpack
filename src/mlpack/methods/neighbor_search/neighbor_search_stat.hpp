/**
 * @file neighbor_search.hpp
 * @author Ryan Curtin
 *
 * Defines the NeighborSearch class, which performs an abstract
 * nearest-neighbor-like query on two datasets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_STAT_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_STAT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace neighbor {

/**
 * Extra data for each node in the tree.  For neighbor searches, each node only
 * needs to store a bound on neighbor distances.
 */
template<typename SortPolicy>
class NeighborSearchStat
{
 private:
  //! The first bound on the node's neighbor distances (B_1).  This represents
  //! the worst candidate distance of any descendants of this node.
  double firstBound;
  //! The second bound on the node's neighbor distances (B_2).  This represents
  //! a bound on the worst distance of any descendants of this node assembled
  //! using the best descendant candidate distance modified by the furthest
  //! descendant distance.
  double secondBound;
  //! The aux bound on the node's neighbor distances (B_aux). This represents
  //! the best descendant candidate distance (used to calculate secondBound).
  double auxBound;
  //! The last distance evaluation.
  double lastDistance;

 public:
  /**
   * Initialize the statistic with the worst possible distance according to
   * our sorting policy.
   */
  NeighborSearchStat() :
      firstBound(SortPolicy::WorstDistance()),
      secondBound(SortPolicy::WorstDistance()),
      auxBound(SortPolicy::WorstDistance()),
      lastDistance(0.0) { }

  /**
   * Initialization for a fully initialized node.  In this case, we don't need
   * to worry about the node.
   */
  template<typename TreeType>
  NeighborSearchStat(TreeType& /* node */) :
      firstBound(SortPolicy::WorstDistance()),
      secondBound(SortPolicy::WorstDistance()),
      auxBound(SortPolicy::WorstDistance()),
      lastDistance(0.0) { }

  /**
   * Reset statistic parameters to initial values.
   */
  void Reset()
  {
    firstBound = SortPolicy::WorstDistance();
    secondBound = SortPolicy::WorstDistance();
    auxBound = SortPolicy::WorstDistance();
    lastDistance = 0.0;
  }

  //! Get the first bound.
  double FirstBound() const { return firstBound; }
  //! Modify the first bound.
  double& FirstBound() { return firstBound; }
  //! Get the second bound.
  double SecondBound() const { return secondBound; }
  //! Modify the second bound.
  double& SecondBound() { return secondBound; }
  //! Get the aux bound.
  double AuxBound() const { return auxBound; }
  //! Modify the aux bound.
  double& AuxBound() { return auxBound; }
  //! Get the last distance calculation.
  double LastDistance() const { return lastDistance; }
  //! Modify the last distance calculation.
  double& LastDistance() { return lastDistance; }

  //! Serialize the statistic to/from an archive.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using data::CreateNVP;

    ar & CreateNVP(firstBound, "firstBound");
    ar & CreateNVP(secondBound, "secondBound");
    ar & CreateNVP(auxBound, "auxBound");
    ar & CreateNVP(lastDistance, "lastDistance");
  }
};

} // namespace neighbor
} // namespace mlpack

#endif
