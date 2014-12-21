/**
 * @file range_search_stat.hpp
 * @author Ryan Curtin
 *
 * Statistic class for RangeSearch, which just holds the last visited node and
 * the corresponding base case result.
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
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_STAT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace range {

/**
 * Statistic class for RangeSearch, to be set to the StatisticType of the tree
 * type that range search is being performed with.  This class just holds the
 * last visited node and the corresponding base case result.
 */
class RangeSearchStat
{
 public:
  /**
   * Initialize the statistic.
   */
  RangeSearchStat() : lastDistanceNode(NULL), lastDistance(0.0) { }

  /**
   * Initialize the statistic given a tree node that this statistic belongs to.
   * In this case, we ignore the node.
   */
  template<typename TreeType>
  RangeSearchStat(TreeType& /* node */) :
      lastDistanceNode(NULL),
      lastDistance(0.0) { }

  //! Get the last distance evaluation node.
  void* LastDistanceNode() const { return lastDistanceNode; }
  //! Modify the last distance evaluation node.
  void*& LastDistanceNode() { return lastDistanceNode; }
  //! Get the last distance evaluation.
  double LastDistance() const { return lastDistance; }
  //! Modify the last distance evaluation.
  double& LastDistance() { return lastDistance; }

 private:
  //! The last distance evaluation node.
  void* lastDistanceNode;
  //! The last distance evaluation.
  double lastDistance;
};

}; // namespace neighbor
}; // namespace mlpack

#endif
