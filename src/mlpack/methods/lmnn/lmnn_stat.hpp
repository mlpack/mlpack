/**
 * @file lmnn_stat.hpp
 * @author Ryan Curtin
 *
 * Defines the LMNNStat class, which holds useful information for LMNN searches
 * in a tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_LMNN_STAT_HPP
#define MLPACK_METHODS_LMNN_LMNN_STAT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace lmnn {

/**
 * Extra data for each node in the tree.  For LMNN searches, each node
 * needs to store a bound on neighbor distances, and also whether or not any
 * impostors or true neighbors are descendants of the node.
 */
class LMNNStat
{
 private:
  //! The first bound on the node's neighbor distances (B_1).  This represents
  //! the worst candidate distance of any descendants of this node.
  double bound;
  //! The last distance evaluation.
  double lastDistance;
  //! Whether or not the node contains any impostors for a given class as
  //! descendant points.
  std::vector<bool> hasImpostors;
  //! Whether or not the node contains any true neighbors for a given class as
  //! descendant points.
  std::vector<bool> hasTrueNeighbors;

 public:
  /**
   * Initialize the statistic with the worst possible bounds.  Note that after
   * construction, hasImpostors and hasTrueNeighbors are still not set!  This
   * must be done after tree building.
   */
  LMNNStat() : bound(DBL_MAX), lastDistance(0.0) { }

  /**
   * Initialization for a fully initialized node.  In this case, we don't need
   * to worry about the node.  Note that after construction, hasImpostors and
   * hasTrueNeighbors are still not set!  This must be done after tree building.
   */
  template<typename TreeType>
  LMNNStat(TreeType& /* node */) : bound(DBL_MAX), lastDistance(0.0) { }

  /**
   * Reset statistic parameters to initial values.
   */
  void Reset()
  {
    bound = DBL_MAX;
    lastDistance = 0.0;
  }

  //! Get the first bound.
  double Bound() const { return bound; }
  //! Modify the first bound.
  double& Bound() { return bound; }
  //! Get the last distance calculation.
  double LastDistance() const { return lastDistance; }
  //! Modify the last distance calculation.
  double& LastDistance() { return lastDistance; }

  //! Get whether or not the node has any impostors as descendants for each
  //! class.
  const std::vector<bool>& HasImpostors() const { return hasImpostors; }
  //! Modify whether or not the node has any impostors as descendants for each
  //! class.
  std::vector<bool>& HasImpostors() { return hasImpostors; }

  //! Get whether or not the node has any true neighbors as descendants for each
  //! class.
  const std::vector<bool>& HasTrueNeighbors() const { return hasTrueNeighbors; }
  //! Modify whether or not the node has any true neighbors as descendants for
  //! each class.
  std::vector<bool>& HasTrueNeighbors() { return hasTrueNeighbors; }

  //! Serialize the statistic to/from an archive.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(bound);
    ar & BOOST_SERIALIZATION_NVP(lastDistance);
    ar & BOOST_SERIALIZATION_NVP(hasImpostors);
    ar & BOOST_SERIALIZATION_NVP(hasTrueNeighbors);
  }
};

} // namespace lmnn
} // namespace mlpack

#endif
