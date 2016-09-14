/**
 * @file octree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of generalized octree (Octree).
 */
#ifndef MLPACK_CORE_TREE_OCTREE_OCTREE_IMPL_HPP
#define MLPACK_CORE_TREE_OCTREE_OCTREE_IMPL_HPP

#include "octree.hpp"

//! Construct the tree.
template<typename MetricType, typename StatisticType, typename MatType>
Octree<MetricType, StatisticType, MatType>::Octree(const MatType& dataset,
                                                   const double maxLeafSize) :
    dataset(new MatType(dataset)),

{
  // Calculate empirical center of data.
  bound |= *dataset;
  arma::vec center = bound.Center();
  double maxWidth = bound.MaxWidth();

  SplitNode(center, maxWidth);

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Construct the tree.
template<typename MetricType, typename StatisticType, typename MatType>
Octree<MetricType, StatisticType, MatType>::Octree(
    const MatType& dataset,
    std::vector<size_t>& oldFromNew,
    const size_t maxLeafSize) :
    dataset(new MatType(dataset)),

{
  // Calculate empirical center of data.
  bound |= *dataset;
  arma::vec center = bound.Center();
  double maxWidth = bound.MaxWidth();

  oldFromNew.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    oldFromNew[i] = i;

  SplitNode(center, maxWidth, oldFromNew);

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Construct the tree.
template<typename MetricType, typename StatisticType, typename MatType>
Octree<MetricType, StatisticType, MatType>::Octree(
    const MatType& dataset,
    std::vector<size_t>& oldFromNew,
    const size_t maxLeafSize) :
    dataset(new MatType(dataset)),

{
  // Calculate empirical center of data.
  bound |= *dataset;
  arma::vec center = bound.Center();
  double maxWidth = bound.MaxWidth();

  oldFromNew.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    oldFromNew[i] = i;

  SplitNode(center, maxWidth, oldFromNew);

  // Initialize the statistic.
  stat = StatisticType(*this);
}


//! Split the node.
template<typename MetricType, typename StatisticType, typename MatType>
void Octree<MetricType, StatisticType, MatType>::SplitNode(
    const arma::vec& center,
    const double width)
{
  // We must split the dataset by sequentially creating each of the children.
  // We do this in two steps: first we make a pass to count the number of points
  // that will fall into each child; then in the second pass we rearrange the
  // points and create the children.
  arma::Col<size_t> childCounts(std::pow(2, dataset->n_rows),
      arma::fill::zeros);
  arma::Col<size_t> assignments(count, arma::fill::zeros);

  // First pass: calculate number of points in each child, and find child
  // assignments for each point.
  for (size_t i = 0; i < count; ++i)
  {
    for (size_t d = 0; d < dataset->n_rows; ++d)
    {
      // We are guaranteed that the points fall within 'width / 2' of the center
      // in each dimension, so we just need to check which side of the center
      // the points fall on.  The last dimension represents the most significant
      // bit in the assignment; the bit is '1' if it falls to the right of the
      // center.
      if (dataset(d, begin + i) > center(d))
        assignments(i) |= (1 << d);
    }

    childCounts(assignments(i))++;
  }

  // Sort all of the points so we know where to copy them.
  arma::uvec ordering = arma::stable_sort_index(assignments, "ascend");

  // This strategy may copy the matrix during the computation, but that isn't
  // really a problem.  We use non-contiguous submatrix views to extract the
  // columns in the correct order.
  dataset->cols(begin, begin + count - 1) = dataset->cols(begin + ordering);

  // Now that the dataset is reordered, we can create the children.
  size_t childBegin = begin;
  arma::vec childCenter(center.n_elem);
  const double childWidth = width / 2.0;
  for (size_t i = 0; i < childCounts.n_elem; ++i)
  {
    // If the child has no points, don't create it.
    if (childCounts[i] == 0)
      continue;

    // Create the correct center.
    for (size_t d = 0; d < center.n_elem; ++d)
    {
      // Is the dimension "right" (1) or "left" (0)?
      if ((i >> d) & 1 == 0)
        childCenter[d] = center[d] - childWidth;
      else
        childCenter[d] = center[d] + childWidth;
    }

    children.push_back(new Octree(this, childBegin, childCounts[i], childCenter,
        childWidth, maxLeafSize));

    childBegin += childCounts[i];
  }
}

//! Split the node, and store mappings.
template<typename MetricType, typename StatisticType, typename MatType>
void Octree<MetricType, StatisticType, MatType>::SplitNode(
    const arma::vec& center,
    const double width,
    std::vector<size_t>& oldFromNew)
{
  // We must split the dataset by sequentially creating each of the children.
  // We do this in two steps: first we make a pass to count the number of points
  // that will fall into each child; then in the second pass we rearrange the
  // points and create the children.
  arma::Col<size_t> childCounts(std::pow(2, dataset->n_rows),
      arma::fill::zeros);
  arma::Col<size_t> assignments(count, arma::fill::zeros);

  // First pass: calculate number of points in each child, and find child
  // assignments for each point.
  for (size_t i = 0; i < count; ++i)
  {
    for (size_t d = 0; d < dataset->n_rows; ++d)
    {
      // We are guaranteed that the points fall within 'width / 2' of the center
      // in each dimension, so we just need to check which side of the center
      // the points fall on.  The last dimension represents the most significant
      // bit in the assignment; the bit is '1' if it falls to the right of the
      // center.
      if (dataset(d, begin + i) > center(d))
        assignments(i) |= (1 << d);
    }

    childCounts(assignments(i))++;
  }

  // Sort all of the points so we know where to copy them.
  arma::uvec ordering = arma::stable_sort_index(assignments, "ascend");

  // This strategy may copy the matrix during the computation, but that isn't
  // really a problem.  We use non-contiguous submatrix views to extract the
  // columns in the correct order.
  dataset->cols(begin, begin + count - 1) = dataset->cols(begin + ordering);
  for (size_t i = 0; i < count; ++i)
    oldFromNew[ordering[i] + begin] = i + begin;

  // Now that the dataset is reordered, we can create the children.
  size_t childBegin = begin;
  arma::vec childCenter(center.n_elem);
  const double childWidth = width / 2.0;
  for (size_t i = 0; i < childCounts.n_elem; ++i)
  {
    // If the child has no points, don't create it.
    if (childCounts[i] == 0)
      continue;

    // Create the correct center.
    for (size_t d = 0; d < center.n_elem; ++d)
    {
      // Is the dimension "right" (1) or "left" (0)?
      if ((i >> d) & 1 == 0)
        childCenter[d] = center[d] - childWidth;
      else
        childCenter[d] = center[d] + childWidth;
    }

    children.push_back(new Octree(this, childBegin, childCounts[i], oldFromNew,
        childCenter, childWidth, maxLeafSize));

    childBegin += childCounts[i];
  }
}
