/**
 * @file dtb.hpp
 * @author Bill March (march@gatech.edu)
 *
 * DTBStat is the StatisticType used by trees when performing EMST.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_EMST_DTB_STAT_HPP
#define __MLPACK_METHODS_EMST_DTB_STAT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace emst {

/**
 * A statistic for use with MLPACK trees, which stores the upper bound on
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

}; // namespace emst
}; // namespace mlpack

#endif // __MLPACK_METHODS_EMST_DTB_STAT_HPP
