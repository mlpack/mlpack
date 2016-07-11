/**
 * @file vantage_point_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (VantagePointSplit) to split a binary space partition
 * tree according to the median value of the distance to a certain vantage point.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP

#include "vantage_point_split.hpp"
#include <mlpack/core/tree/bounds.hpp>

namespace mlpack {
namespace tree {

template<typename BoundType, typename MatType>
bool VantagePointSplit<BoundType, MatType>::
SplitNode(const BoundType& bound, MatType& data, const size_t begin,
    const size_t count, size_t& splitCol)
{
  typename BoundType::ElemType mu;
  size_t vantagePointIndex;

  // Find the best vantage point
  SelectVantagePoint(bound, data, begin, count, vantagePointIndex, mu);

  // All points are equal
  if (mu == 0)
    return false;

  data.swap_cols(begin, vantagePointIndex);

  arma::Col<ElemType> vantagePoint = data.col(begin);
  splitCol = PerformSplit(bound, data, begin, count, vantagePoint, mu);

  assert(splitCol > begin);
  assert(splitCol < begin + count);
  return true;
}

template<typename BoundType, typename MatType>
bool VantagePointSplit<BoundType, MatType>::
SplitNode(const BoundType& bound, MatType& data, const size_t begin,
    const size_t count, size_t& splitCol, std::vector<size_t>& oldFromNew)
{
  ElemType mu;
  size_t vantagePointIndex;

  // Find the best vantage point
  SelectVantagePoint(bound, data, begin, count, vantagePointIndex, mu);

  // All points are equal
  if (mu == 0)
    return false;

  data.swap_cols(begin, vantagePointIndex);
  size_t t = oldFromNew[begin];
  oldFromNew[begin] = oldFromNew[vantagePointIndex];
  oldFromNew[vantagePointIndex] = t;

  arma::Col<ElemType> vantagePoint = data.col(begin);

  splitCol = PerformSplit(bound, data, begin, count, vantagePoint, mu, oldFromNew);

  assert(splitCol > begin);
  assert(splitCol < begin + count);
  return true;
}

template<typename BoundType, typename MatType>
template <typename VecType>
size_t VantagePointSplit<BoundType, MatType>::PerformSplit(const BoundType& bound,
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
  while (AssignToLeftSubtree(bound, data, vantagePoint, left, mu) && (left <= right))
    left++;

  while ((!AssignToLeftSubtree(bound, data, vantagePoint, right, mu)) && (left <= right) && (right > 0))
    right--;

  while (left <= right)
  {
    // Swap columns.
    data.swap_cols(left, right);

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while ((AssignToLeftSubtree(bound, data, vantagePoint, left, mu)) && (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((!AssignToLeftSubtree(bound, data, vantagePoint, right, mu)) && (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);

  return left;
}

template<typename BoundType, typename MatType>
template<typename VecType>
size_t VantagePointSplit<BoundType, MatType>::PerformSplit(const BoundType& bound,
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

  while (AssignToLeftSubtree(bound, data, vantagePoint, left, mu) && (left <= right))
    left++;

  while ((!AssignToLeftSubtree(bound, data, vantagePoint, right, mu)) && (left <= right) && (right > 0))
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
    while (AssignToLeftSubtree(bound, data, vantagePoint, left, mu) && (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((!AssignToLeftSubtree(bound, data, vantagePoint, right, mu)) && (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);

  return left;
}

template<typename BoundType, typename MatType>
void VantagePointSplit<BoundType, MatType>::
SelectVantagePoint(const BoundType& bound, const MatType& data,
    const size_t begin, const size_t count, size_t& vantagePoint, ElemType& mu)
{
  arma::uvec vantagePointCandidates;

  // Get no more than max(maxNumSamples, count) vantage point candidates
  GetDistinctSamples(vantagePointCandidates, maxNumSamples, begin, count);

  ElemType bestSpread = 0;

  //  Evaluate eache candidate
  for (size_t i = 0; i < vantagePointCandidates.n_rows; i++)
  {
    arma::uvec samples;

    // Get no more than max(maxNumSamples, count) random samples
    GetDistinctSamples(samples, maxNumSamples, begin, count);

    // Calculate the second moment of the distance to the vantage point candidate
    // using these random samples
    const ElemType spread = GetSecondMoment(bound, data, samples,
        vantagePointCandidates[i]);

    if (spread > bestSpread)
    {
      bestSpread = spread;
      vantagePoint = vantagePointCandidates[i];
      //  Calculate the median value of the distance from the vantage point candidate
      //  to these samples
      GetMedian(bound, data, samples, vantagePoint, mu);
    }    
  }
  assert(bestSpread > 0);
}

template<typename BoundType, typename MatType>
void VantagePointSplit<BoundType, MatType>::
GetDistinctSamples(arma::uvec& distinctSamples, const size_t numSamples,
    const size_t begin, const size_t upperBound)
{
  if (upperBound > numSamples)
  {
    arma::Col<size_t> samples;

    samples.zeros(upperBound);

    for (size_t i = 0; i < numSamples; i++)
      samples [ (size_t) math::RandInt(upperBound) ]++;

    distinctSamples = arma::find(samples > 0);

    distinctSamples += begin;
  }
  else
  {
    //  The node contains less points than requested
    distinctSamples.set_size(upperBound);
    for (size_t i = 0; i < upperBound; i++)
      distinctSamples[i] = begin + i;
  }
}

template<typename BoundType, typename MatType>
void VantagePointSplit<BoundType, MatType>::
GetMedian(const BoundType& bound, const MatType& data,
    const arma::uvec& samples, const size_t vantagePoint, ElemType& mu)
{
  std::vector<SortStruct<ElemType>> sorted(samples.n_rows);

  for (size_t i = 0; i < samples.n_rows; i++)
  {
    sorted[i].point = samples[i];
    sorted[i].dist = bound.Metric().Evaluate(data.col(vantagePoint),
        data.col(samples[i]));
  }

  //  Sort samples according to the distance to the vantage point
  std::sort(sorted.begin(), sorted.end(), StructComp<ElemType>);

  //  Get the midian value
  mu = bound.Metric().Evaluate(data.col(vantagePoint),
      data.col(sorted[sorted.size() / 2].point));
}

template<typename BoundType, typename MatType>
typename MatType::elem_type VantagePointSplit<BoundType, MatType>::
GetSecondMoment(const BoundType& bound, const MatType& data,
    const arma::uvec& samples,  const size_t vantagePoint)
{
  ElemType moment = 0;

  for (size_t i = 0; i < samples.size(); i++)
  {
    const ElemType dist =
        bound.Metric().Evaluate(data.col(vantagePoint), data.col(samples[i]));

    moment += dist * dist;
  }

  moment /= samples.size();

  return moment;
}

template<typename BoundType, typename MatType>
template<typename VecType>
bool VantagePointSplit<BoundType, MatType>::
AssignToLeftSubtree(const BoundType& bound, const MatType& mat,
    const VecType& vantagePoint, const size_t point, const ElemType mu)
{
  // Return true if the point is close to the vantage point
  if (bound.Metric().Evaluate(vantagePoint, mat.col(point)) < mu)
    return true;

  return false;
}

} // namespace tree
} // namespace mlpack

#endif  // MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP
