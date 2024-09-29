/**
 * @file methods/dbscan/ordered_point_selection.hpp
 * @author Kim SangYeon
 *
 * Sequentially select the next point for DBSCAN.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DBSCAN_ORDERED_POINT_SELECTION_HPP
#define MLPACK_METHODS_DBSCAN_ORDERED_POINT_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class can be used to sequentially select the next point to use for DBSCAN.
 */
class OrderedPointSelection
{
 public:
  /**
   * Select the next point to use, sequentially.
   *
   * @param point unvisited Bitset indicating which points are unvisited.
   * @param * (data) Unused data.
   */
  template<typename MatType>
  static size_t Select(const size_t point,
                       const MatType& /* data */)
  {
    return point; // Just return point.
  }
};

} // namespace mlpack

#endif
