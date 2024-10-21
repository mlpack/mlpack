/**
 * @file build_tree.hpp
 * @author Ryan Curtin
 *
 * Auxiliary functions to build trees with and without meetings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BUILD_TREE_HPP
#define MLPACK_CORE_TREE_BUILD_TREE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

//! Construct tree that rearranges the dataset.
template<typename TreeType, typename MatType>
TreeType* BuildTree(
    MatType&& dataset,
    std::vector<size_t>& oldFromNew,
    const std::enable_if_t<TreeTraits<TreeType>::RearrangesDataset>* = 0)
{
  return new TreeType(std::forward<MatType>(dataset), oldFromNew);
}

//! Construct tree that doesn't rearrange the dataset.
template<typename TreeType, typename MatType>
TreeType* BuildTree(
    MatType&& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const std::enable_if_t<!TreeTraits<TreeType>::RearrangesDataset>* = 0)
{
  return new TreeType(std::forward<MatType>(dataset));
}

} // namespace mlpack

#endif
