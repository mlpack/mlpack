/**
 * @file perform_split.hpp
 * @author Mikhail Lozhnikov
 *
 * This file contains functions that implement the default binary split
 * behavior. The functions perform the actual splitting. This will order
 * the dataset such that points that belong to the left subtree are on the left
 * of the split column, and points from the right subtree are on the right side
 * of the split column.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_PERFORM_SPLIT_HPP
#define MLPACK_CORE_TREE_PERFORM_SPLIT_HPP

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {
namespace split {

/**
 * This function implements the default split behavior i.e. it rearranges
 * points according to the split information. The SplitType::AssignToLeftNode()
 * function is used in order to determine the child that contains any particular
 * point.
 *
 * @param data The dataset used by the binary space tree.
 * @param begin Index of the starting point in the dataset that belongs to
 *    this node.
 * @param count Number of points in this node.
 * @param splitInfo The information about the split.
 */
template<typename MatType, typename SplitType>
size_t PerformSplit(MatType& data,
                    const size_t begin,
                    const size_t count,
                    const typename SplitType::SplitInfo& splitInfo)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.
  size_t left = begin;
  size_t right = begin + count - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while ((left <= right) &&
      (SplitType::AssignToLeftNode(data.col(left), splitInfo)))
    left++;
  while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
      (left <= right) && (right > 0))
    right--;

  // Shortcut for when all points are on the right.
  if (left == right && right == 0)
    return left;

  while (left <= right)
  {
    // Swap columns.
    data.swap_cols(left, right);

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while (SplitType::AssignToLeftNode(data.col(left), splitInfo) &&
        (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
        (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);

  return left;
}

/**
 * This function implements the default split behavior i.e. it rearranges
 * points according to the split information. The SplitType::AssignToLeftNode()
 * function is used in order to determine the child that contains any particular
 * point. The function takes care of indices and returns the list of changed
 * indices.
 *
 * @param data The dataset used by the binary space tree.
 * @param begin Index of the starting point in the dataset that belongs to
 *    this node.
 * @param count Number of points in this node.
 * @param splitInfo The information about the split.
 * @param oldFromNew Vector which will be filled with the old positions for
 *    each new point.
 */
template<typename MatType, typename SplitType>
size_t PerformSplit(MatType& data,
                    const size_t begin,
                    const size_t count,
                    const typename SplitType::SplitInfo& splitInfo,
                    std::vector<size_t>& oldFromNew)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.
  size_t left = begin;
  size_t right = begin + count - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while ((left <= right) &&
         (SplitType::AssignToLeftNode(data.col(left), splitInfo)))
    left++;
  while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
         (left <= right) && (right > 0))
    right--;

  // Shortcut for when all points are on the right.
  if (left == right && right == 0)
    return left;

  while (left <= right)
  {
    // Swap columns.
    data.swap_cols(left, right);

    // Update the indices for what we changed.
    size_t t = oldFromNew[left];
    oldFromNew[left] = oldFromNew[right];
    oldFromNew[right] = t;

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while (SplitType::AssignToLeftNode(data.col(left), splitInfo) &&
        (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((!SplitType::AssignToLeftNode(data.col(right), splitInfo)) &&
        (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);
  return left;
}

} // namespace split
} // namespace tree
} // namespace mlpack


#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_PERFORM_SPLIT_HPP
