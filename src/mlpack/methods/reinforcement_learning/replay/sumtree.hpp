/**
 * @file sumtree.hpp
 * @author Xiaohong
 *
 * This file is an implementation of sum tree.
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
  template<typename T>
  class SumTree {
    SumTree(size_t capacity):
        capacity(capacity)
    {
      auxiliary(capacity, 0);
      element(capacity, 0);
    }

    void set(size_t idx, T value)
    {
//     btodo: update the leaf node value
    }

    T get(size_t)
    {
//     btodo: get the leaf node value
      return 0;
    }

    T sum(size_t _start, size_t _end)
    {
//      btodo: caculate the sum of contiguous subsequence of the array.
      return 0;
    }

    size_t findPrefixSum(T mass)
    {
//      btodo: Find the highest index `idx` in the array such that
//            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
    }

   private:
    capacity;
    std::vector<T> auxiliary;
    std::vector<T> element;
  };
} // namespace rl
} // namespace mlpack

#endif
