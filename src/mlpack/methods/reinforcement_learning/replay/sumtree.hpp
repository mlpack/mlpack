/**
 * @file sumtree.hpp
 * @author Xiaohong
 *
 * This file is an implementation of sumtree.
 *
 * reference:
 * [1] https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
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
namespace rl {

/**
 * Implementation of SumTree.
 */

template<typename T>
class SumTree
{
 public:
  SumTree(size_t capacity):
      capacity(capacity)
  {
    element = std::vector<T> (2 * capacity);
  }

  void set(size_t idx, T value)
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

  T get(size_t idx)
  {
    idx += capacity;
    return element[idx];
  }

  T sumHelper(size_t _start, size_t _end, size_t node, size_t node_start, size_t node_end)
  {
    if (_start == node_start && _end == node_end)
    {
      return element[node];
    }
    size_t mid = (node_start + node_end) / 2;
    if (_end <= mid)
    {
      return sumHelper(_start, _end, 2 * node, node_start, mid);
    }
    else
    {
      if (mid + 1 <= _start)
      {
        return sumHelper(_start, _end, 2 * node + 1, mid + 1 , node_end);
      }
      else
      {
        return sumHelper(_start, mid, 2 * node, node_start, mid) +
                sumHelper(mid+1, _end, 2 * node + 1, mid + 1 , node_end);
      }
    }
  }

  T sum(size_t _start, size_t _end)
  {
//  caculate the sum of contiguous subsequence of the array.
    _end -= 1;
    return sumHelper(_start, _end, 1, 0, capacity-1);
  }

  T sum()
  {
    return sum(0, capacity);
  }

  size_t findPrefixSum(T mass)
  {
//    Find the highest index `idx` in the array such that
//            sum(arr[0] + arr[1] + ... + arr[i]) <= mass
    int idx = 1;
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
  size_t capacity;
  std::vector<T> element;
};

} // namespace rl
} // namespace mlpack

#endif
