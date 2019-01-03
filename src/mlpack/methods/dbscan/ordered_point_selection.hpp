/**
 * @file ordered_point_selection.hpp
 * @author Yugandhar Tripathi
 *
 * Select the next point in order for DBSCAN.
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
namespace dbscan {

/**
 * This class can be used to select the next point to use for DBSCAN.
 */
class OrderedPointSelection
{
 public:
  /**
   * Select the next point to use.
   *
   * @param index next point in order.
   * @param Unused data.
   */
  template<typename MatType>
  static size_t Select(const size_t index,
                       const MatType& /*Unused*/)
  {
    return index;
  }
};

} // namespace dbscan
} // namespace mlpack

#endif
