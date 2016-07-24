/**
 * @file vantage_point_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (VantagePointSplit) to split a vantage point
 * tree according to the median value of the distance to a certain vantage point.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP

#include "vantage_point_split.hpp"
#include <mlpack/core/tree/bounds.hpp>

namespace mlpack {
namespace tree {

template<typename BoundType, typename MatType, size_t maxNumSamples>
bool VantagePointSplit<BoundType, MatType, maxNumSamples>::
SplitNode(const BoundType& bound, MatType& data, const size_t begin,
    const size_t count, size_t& splitCol)
{
  ElemType mu = 0;
  size_t vantagePointIndex;

  // Find the best vantage point
  SelectVantagePoint(bound.Metric(), data, begin, count, vantagePointIndex, mu);

  // All points are equal
  if (mu == 0)
    return false;

  // The first point of the left child is centroid.
  data.swap_cols(begin, vantagePointIndex);

  arma::Col<ElemType> vantagePoint = data.col(begin);
  splitCol = PerformSplit(bound.Metric(), data, begin, count, vantagePoint, mu);

  assert(splitCol > begin);
  assert(splitCol < begin + count);
  return true;
}

template<typename BoundType, typename MatType, size_t maxNumSamples>
bool VantagePointSplit<BoundType, MatType, maxNumSamples>::
SplitNode(const BoundType& bound, MatType& data, const size_t begin,
    const size_t count, size_t& splitCol, std::vector<size_t>& oldFromNew)
{
  ElemType mu = 0;
  size_t vantagePointIndex;

  // Find the best vantage point
  SelectVantagePoint(bound.Metric(), data, begin, count, vantagePointIndex, mu);

  // All points are equal
  if (mu == 0)
    return false;

  // The first point of the left child is centroid.
  data.swap_cols(begin, vantagePointIndex);
  size_t t = oldFromNew[begin];
  oldFromNew[begin] = oldFromNew[vantagePointIndex];
  oldFromNew[vantagePointIndex] = t;

  arma::Col<ElemType> vantagePoint = data.col(begin);

  splitCol = PerformSplit(bound.Metric(), data, begin, count, vantagePoint, mu,
      oldFromNew);

  assert(splitCol > begin);
  assert(splitCol < begin + count);
  return true;
}

template<typename BoundType, typename MatType, size_t maxNumSamples>
template <typename VecType>
size_t VantagePointSplit<BoundType, MatType, maxNumSamples>::PerformSplit(
    const MetricType& metric,
    MatType& data,
    const size_t begin,
    const size_t count,
    const VecType& vantagePoint,
    const ElemType mu)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points closer to
  // the vantage point should be on the left side of the matrix, and the farther
  // from the vantage point should be on the right side of the matrix.
  size_t left = begin;
  size_t right = begin + count - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while (AssignToLeftSubtree(metric, data, vantagePoint, left, mu) &&
      (left <= right))
    left++;

  while ((!AssignToLeftSubtree(metric, data, vantagePoint, right, mu)) &&
      (left <= right) && (right > 0))
    right--;

  while (left <= right)
  {
    // Swap columns.
    data.swap_cols(left, right);

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while ((AssignToLeftSubtree(metric, data, vantagePoint, left, mu)) &&
        (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((!AssignToLeftSubtree(metric, data, vantagePoint, right, mu)) &&
        (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);

  return left;
}

template<typename BoundType, typename MatType, size_t maxNumSamples>
template<typename VecType>
size_t VantagePointSplit<BoundType, MatType, maxNumSamples>::PerformSplit(
    const MetricType& metric,
    MatType& data,
    const size_t begin,
    const size_t count,
    const VecType& vantagePoint,
    const ElemType mu,
    std::vector<size_t>& oldFromNew)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points closer to
  // the vantage point should be on the left side of the matrix, and the farther
  // from the vantage point should be on the right side of the matrix.
  size_t left = begin;
  size_t right = begin + count - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.

  while (AssignToLeftSubtree(metric, data, vantagePoint, left, mu) &&
      (left <= right))
    left++;

  while ((!AssignToLeftSubtree(metric, data, vantagePoint, right, mu)) &&
      (left <= right) && (right > 0))
    right--;

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
    while (AssignToLeftSubtree(metric, data, vantagePoint, left, mu) &&
        (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((!AssignToLeftSubtree(metric, data, vantagePoint, right, mu)) &&
        (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);

  return left;
}

template<typename BoundType, typename MatType, size_t maxNumSamples>
void VantagePointSplit<BoundType, MatType, maxNumSamples>::
SelectVantagePoint(const MetricType& metric, const MatType& data,
    const size_t begin, const size_t count, size_t& vantagePoint, ElemType& mu)
{
  arma::uvec vantagePointCandidates;
  arma::Col<ElemType> distances(maxNumSamples);

  // Get no more than max(maxNumSamples, count) vantage point candidates
  math::ObtainDistinctSamples(begin, begin + count, maxNumSamples,
      vantagePointCandidates);

  ElemType bestSpread = 0;

  arma::uvec samples;
  //  Evaluate each candidate
  for (size_t i = 0; i < vantagePointCandidates.n_elem; i++)
  {
    // Get no more than min(maxNumSamples, count) random samples
    math::ObtainDistinctSamples(begin, begin + count, maxNumSamples, samples);

    // Calculate the second moment of the distance to the vantage point candidate
    // using these random samples
    distances.set_size(samples.n_elem);

    for (size_t j = 0; j < samples.n_elem; j++)
      distances[j] = metric.Evaluate(data.col(vantagePointCandidates[i]),
        data.col(samples[j]));

    const ElemType spread = arma::sum(distances % distances) / samples.n_elem;

    if (spread > bestSpread)
    {
      bestSpread = spread;
      vantagePoint = vantagePointCandidates[i];
      //  Calculate the median value of the distance from the vantage point candidate
      //  to these samples
      mu = arma::median(distances);
   }    
  }
  assert(bestSpread > 0);
}

} // namespace tree
} // namespace mlpack

#endif  // MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP
