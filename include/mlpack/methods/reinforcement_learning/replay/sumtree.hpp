/**
 * @file methods/reinforcement_learning/replay/sumtree.hpp
 * @author Xiaohong
 *
 * This file is an implementation of sumtree. Based on:
 * https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_SUMTREE_HPP
#define MLPACK_METHODS_RL_SUMTREE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of SumTree.
 *
 * Build a Segment Tree like data structure.
 * https://en.wikipedia.org/wiki/Segment_tree
 *
 * Used to maintain prefix-sum of an array.
 *
 * @tparam T The array's element type.
 */
template<typename T>
class SumTree
{
 public:
  /**
   * Default constructor.
   */
  SumTree() : capacity(0)
  { /* Nothing to do here. */ }

  /**
   * Construct an instance of SumTree class.
   *
   * @param capacity Size of data.
   */
  SumTree(const size_t capacity) : capacity(capacity)
  {
    element = std::vector<T>(2 * capacity);
  }

  /**
   * Set the data array with idx.
   *
   * @param idx The array idx to be changed.
   * @param value The data that array with idx to be.
   */
  void Set(size_t idx, const T value)
  {
    idx += capacity;
    element[idx] = value;
    idx /= 2;
    while (idx >= 1)
    {
      element[idx] = element[2 * idx] + element[2 * idx + 1];
      idx /= 2;
    }
  }

  /**
   * Update the data with batch rather loop over the indices with set method.
   *
   * @param indices The indices of data to be changed.
   * @param data The data that array with indices to be.
   */
  void BatchUpdate(const arma::ucolvec& indices, const arma::Col<T>& data)
  {
    for (size_t i = 0; i < indices.n_rows; ++i)
    {
      element[indices[i] + capacity] = data[i];
    }
    // update the total tree with bottom-up technique.
    for (size_t i = capacity - 1; i > 0; i--)
    {
      element[i] = element[2 * i] + element[2 * i + 1];
    }
  }

  /**
   * Get the data array with idx.
   *
   * @param idx The array idx to get data.
   */
  T Get(size_t idx)
  {
    idx += capacity;
    return element[idx];
  }

  /**
   * Help function for the `sum` function
   *
   * @param start The starting position of subsequence.
   * @param end The end position of subsequence.
   * @param node Reference position.
   * @param nodeStart Starting position of reference segment.
   * @param nodeEnd End position of reference segment.
   */
  T SumHelper(const size_t start,
              const size_t end,
              const size_t node,
              const size_t nodeStart,
              const size_t nodeEnd)
  {
    if (start == nodeStart && end == nodeEnd)
    {
      return element[node];
    }
    size_t mid = (nodeStart + nodeEnd) / 2;
    if (end <= mid)
    {
      return SumHelper(start, end, 2 * node, nodeStart, mid);
    }
    else
    {
      if (mid + 1 <= start)
      {
        return SumHelper(start, end, 2 * node + 1, mid + 1 , nodeEnd);
      }
      else
      {
        return SumHelper(start, mid, 2 * node, nodeStart, mid) +
            SumHelper(mid + 1, end, 2 * node + 1, mid + 1 , nodeEnd);
      }
    }
  }

  /**
   * Calculate the sum of contiguous subsequence of the array.
   *
   * @param start The starting position of subsequence.
   * @param end The end position of subsequence.
   */
  T Sum(const size_t start, size_t end)
  {
    end -= 1;
    return SumHelper(start, end, 1, 0, capacity - 1);
  }

  /**
   * Shortcut for calculating the sum of whole array.
   */
  T Sum()
  {
    return Sum(0, capacity);
  }

  /**
   * Find the highest index `idx` in the array such that
   * sum(arr[0] + arr[1] + ... + arr[idx]) <= mass.
   *
   * @param mass The upper bound of segment array sum.
   */
  size_t FindPrefixSum(T mass)
  {
    size_t idx = 1;
    while (idx < capacity)
    {
      if (element[2 * idx] > mass)
      {
        idx = 2 * idx;
      }
      else
      {
        mass -= element[2 * idx];
        idx = 2 * idx + 1;
      }
    }
    return idx - capacity;
  }

 private:
  //! The capacity of the data array.
  size_t capacity;

  //! Double size of capacity, maintain the segment sum of data.
  std::vector<T> element;
};

} // namespace mlpack

#endif
