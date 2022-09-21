/**
 * @file methods/emst/dtb_stat.hpp
 * @author Bill March (march@gatech.edu)
 *
 * DTBStat is the StatisticType used by trees when performing EMST.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_EMST_DTB_STAT_HPP
#define MLPACK_METHODS_EMST_DTB_STAT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A statistic for use with mlpack trees, which stores the upper bound on
 * distance to nearest neighbors and the component which this node belongs to.
 */
class DTBStat
{
 private:
  //! Upper bound on the distance to the nearest neighbor of any point in this
  //! node.
  double maxNeighborDistance;

  //! Lower bound on the distance to the nearest neighbor of any point in this
  //! node.
  double minNeighborDistance;

  //! Total bound for pruning.
  double bound;

  //! The index of the component that all points in this node belong to.  This
  //! is the same index returned by UnionFind for all points in this node.  If
  //! points in this node are in different components, this value will be
  //! negative.
  int componentMembership;

 public:
  /**
   * A generic initializer.  Sets the maximum neighbor distance to its default,
   * and the component membership to -1 (no component).
   */
  DTBStat() :
      maxNeighborDistance(DBL_MAX),
      minNeighborDistance(DBL_MAX),
      bound(DBL_MAX),
      componentMembership(-1) { }

  /**
   * This is called when a node is finished initializing.  We set the maximum
   * neighbor distance to its default, and if possible, we set the component
   * membership of the node (if it has only one point and no children).
   *
   * @param node Node that has been finished.
   */
  template<typename TreeType>
  DTBStat(const TreeType& node) :
      maxNeighborDistance(DBL_MAX),
      minNeighborDistance(DBL_MAX),
      bound(DBL_MAX),
      componentMembership(
          ((node.NumPoints() == 1) && (node.NumChildren() == 0)) ?
            node.Point(0) : -1) { }

  //! Get the maximum neighbor distance.
  double MaxNeighborDistance() const { return maxNeighborDistance; }
  //! Modify the maximum neighbor distance.
  double& MaxNeighborDistance() { return maxNeighborDistance; }

  //! Get the minimum neighbor distance.
  double MinNeighborDistance() const { return minNeighborDistance; }
  //! Modify the minimum neighbor distance.
  double& MinNeighborDistance() { return minNeighborDistance; }

  //! Get the total bound for pruning.
  double Bound() const { return bound; }
  //! Modify the total bound for pruning.
  double& Bound() { return bound; }

  //! Get the component membership of this node.
  int ComponentMembership() const { return componentMembership; }
  //! Modify the component membership of this node.
  int& ComponentMembership() { return componentMembership; }
}; // class DTBStat

} // namespace mlpack

#endif // MLPACK_METHODS_EMST_DTB_STAT_HPP
