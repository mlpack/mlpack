/**
 * @file core/tree/binary_space_tree/midpoint_split_impl.hpp
 * @author Yash Vadalia
 * @author Ryan Curtin
 *
 * Implementation of class (MidpointSplit) to split a binary space partition
 * tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_MIDPOINT_SPLIT_IMPL_HPP

#include "midpoint_split.hpp"
#include <mlpack/core/tree/bounds.hpp>

namespace mlpack {

template<typename BoundType, typename MatType>
bool MidpointSplit<BoundType, MatType>::SplitNode(const BoundType& bound,
                                                  MatType& data,
                                                  const size_t begin,
                                                  const size_t count,
                                                  SplitInfo& splitInfo)
{
  double maxWidth = -1;
  splitInfo.splitDimension = data.n_rows; // Indicate invalid.

  // Find the split dimension.  If the bound is tight, we only need to consult
  // the bound's width.
  if (BoundTraits<BoundType>::HasTightBounds)
  {
    for (size_t d = 0; d < data.n_rows; d++)
    {
      const double width = bound[d].Width();

      if (width > maxWidth)
      {
        maxWidth = width;
        splitInfo.splitDimension = d;

        // Split in the midpoint of that dimension.
        splitInfo.splitVal = bound[d].Mid();
      }
    }
  }
  else
  {
    // We must individually calculate bounding boxes.
    Range* ranges = new Range[data.n_rows];
    for (size_t i = begin; i < begin + count; ++i)
    {
      // Expand each dimension as necessary.
      for (size_t d = 0; d < data.n_rows; ++d)
      {
        const double val = data(d, i);
        if (val < ranges[d].Lo())
          ranges[d].Lo() = val;
        if (val > ranges[d].Hi())
          ranges[d].Hi() = val;
      }
    }

    // Now, which is the widest?
    for (size_t d = 0; d < data.n_rows; d++)
    {
      const double width = ranges[d].Width();
      if (width > maxWidth)
      {
        maxWidth = width;
        splitInfo.splitDimension = d;
        // Split in the midpoint of that dimension.
        splitInfo.splitVal = ranges[d].Mid();
      }
    }

    delete[] ranges;
  }

  if (maxWidth <= 0) // All these points are the same.  We can't split.
    return false;

  // Split in the midpoint of that dimension.
  splitInfo.splitVal = bound[splitInfo.splitDimension].Mid();

  return true;
}

} // namespace mlpack

#endif
