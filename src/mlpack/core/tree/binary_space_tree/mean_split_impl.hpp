/**
 * @file core/tree/binary_space_tree/mean_split_impl.hpp
 * @author Yash Vadalia
 * @author Ryan Curtin
 *
 * Implementation of class(MeanSplit) to split a binary space partition tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_MEAN_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_MEAN_SPLIT_IMPL_HPP

#include "mean_split.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename BoundType, typename MatType>
bool MeanSplit<BoundType, MatType>::SplitNode(const BoundType& bound,
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
      }
    }

    delete[] ranges;
  }

  if (maxWidth == 0) // All these points are the same.  We can't split.
    return false;

  // Split in the mean of that dimension.
  splitInfo.splitVal = 0.0;
  for (size_t i = begin; i < begin + count; ++i)
    splitInfo.splitVal += data(splitInfo.splitDimension, i);
  splitInfo.splitVal /= count;

  Log::Assert(splitInfo.splitVal >= bound[splitInfo.splitDimension].Lo());
  Log::Assert(splitInfo.splitVal <= bound[splitInfo.splitDimension].Hi());

  return true;
}

} // namespace mlpack

#endif
