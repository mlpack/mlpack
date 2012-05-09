/**
 * @file cover_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of CoverTree class.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_IMPL_HPP

// In case it hasn't already been included.
#include "cover_tree.hpp"

namespace mlpack {
namespace tree {

// Create the cover tree.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
CoverTree<MetricType, RootPointPolicy, StatisticType>::CoverTree(
    const arma::mat& dataset,
    const double expansionConstant) :
    dataset(dataset),
    point(RootPointPolicy::ChooseRoot(dataset)),
    expansionConstant(expansionConstant)
{
  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset.n_cols - 1, dataset.n_cols - 1);
  // This is now [1 2 3 4 ... n].  We must be sure that our point does not
  // occur.
  if (point != 0)
    indices[point - 1] = 0; // Put 0 back into the set; remove what was there.

  arma::vec distances(dataset.n_cols - 1);

  // Build the initial distances.
  ComputeDistances(point, indices, distances, dataset.n_cols - 1);

  // Now determine the scale factor of the root node.
  const double maxDistance = max(distances);
  scale = (int) ceil(log(maxDistance) / log(expansionConstant));
  const double bound = pow(expansionConstant, scale - 1);

  // Unfortunately, we can't call out to other constructors, so we have to copy
  // a little bit of code from the other constructor.  First we build the self
  // child.
  size_t childNearSetSize = SplitNearFar(indices, distances, bound,
      dataset.n_cols - 1);
  size_t childFarSetSize = (dataset.n_cols - 1) - childNearSetSize;
  size_t childUsedSetSize = 0;
  children.push_back(new CoverTree(dataset, expansionConstant, point, scale - 1,
      indices, distances, childNearSetSize, childFarSetSize, childUsedSetSize));

  size_t nearSetSize = (dataset.n_cols - 1) - childUsedSetSize;

  // We have no far set, so the array is organized thusly:
  // [ near | used ].  No resorting is necessary.
  // Therefore, go ahead and build the children.
  arma::Col<size_t> nearSet = indices.rows(0, nearSetSize - 1);
  for (size_t i = 0; i < nearSet.n_elem; ++i)
  {
    // If this point has been used, skip to the next one.
    if (nearSet[i] == dataset.n_cols)
      continue;

    const size_t newPointIndex = nearSet[i]; // nearSet holds indices.

    // We need to move this point into the used set.  To do this we'll swap it
    // with the last value in the far set and then increment the counters
    // accordingly.  We don't have to worry about the fact that the point we
    // swapped is actually in the far set but grouped with the near set, because
    // we're about to rebuild that anyway.
    size_t setIndex;
    for (size_t k = 0; k < nearSetSize; ++k)
      if (indices[k] == newPointIndex)
        setIndex = k;

    // Ensure we need to swap.
    if (setIndex != (nearSetSize - 1))
    {
      // Perform the swap.
      const size_t otherLocation = nearSetSize - 1;
      const double tmpDist = distances[setIndex];

      indices[setIndex] = indices[otherLocation];
      distances[setIndex] = distances[otherLocation];

      indices[otherLocation] = newPointIndex;
      distances[otherLocation] = tmpDist;
    }

    // Update the near set size.  The used set size is updated by the recursive
    // child constructor.
    nearSetSize--;

    // Rebuild the distances for this point.
    ComputeDistances(newPointIndex, indices, distances, nearSetSize);

    // Split into near and far sets for this point.
    childNearSetSize = SplitNearFar(indices, distances, bound, nearSetSize);

    // Build this child (recursively).
    childUsedSetSize = 0;
    childFarSetSize = (nearSetSize - childNearSetSize);
    children.push_back(new CoverTree(dataset, expansionConstant, newPointIndex,
        scale - 1, indices, distances, childNearSetSize, childFarSetSize,
        childUsedSetSize));

    // Now the arrays, in memory, look like this:
    // [ childFar | childUsed | used ]
    // So we don't really need to do anything to them to get ready for the next
    // round.  We do need to look at the points that were used and update the
    // nearSet array.  This double for loop is suboptimal, but the best way I
    // can think of to do this.
    for (size_t j = childFarSetSize; j < (childFarSetSize + childUsedSetSize);
         ++j)
      for (size_t k = i + 1; k < nearSet.n_elem; ++k)
        if (indices[j] == nearSet[k])
          nearSet[k] = dataset.n_cols; // Invalid index to indicate it's used.

    // Now we update the count of near set points.
    nearSetSize -= childUsedSetSize;
  }
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
CoverTree<MetricType, RootPointPolicy, StatisticType>::CoverTree(
    const arma::mat& dataset,
    const double expansionConstant,
    const size_t pointIndex,
    const int scale,
    arma::Col<size_t>& indices,
    arma::vec& distances,
    size_t nearSetSize,
    size_t& farSetSize,
    size_t& usedSetSize) :
    dataset(dataset),
    point(pointIndex),
    scale(scale),
    expansionConstant(expansionConstant)
{

  // If the size of the near set is 0, this is a leaf.
  if (nearSetSize == 0)
    return;

  // Determine the next scale level.  This should be the first level where there
  // are any points in the far set.  So, if we know the maximum distance in the
  // distances array, this will be the largest i such that
  //   maxDistance > pow(ec, i)
  // and using this for the scale factor should guarantee we are not creating an
  // implicit node.  If the maximum distance is 0, every point in the near set
  // will be created as a leaf, and a child to this node.
  const double maxDistance = max(distances.rows(0, nearSetSize - 1));
  if (maxDistance == 0)
  {
    // Make the self child at the lowest possible level.
    // This should not modify farSetSize or usedSetSize.
    children.push_back(new CoverTree(dataset, expansionConstant, pointIndex,
        INT_MIN, indices, distances, 0, farSetSize, usedSetSize));

    // Every point in the near set should be a leaf.
    for (size_t i = 0; i < nearSetSize; ++i)
    {
      // farSetSize and usedSetSize will not be modified.
      children.push_back(new CoverTree(dataset, expansionConstant, indices[i],
          INT_MIN, indices, distances, 0, farSetSize, usedSetSize));
      usedSetSize++;
    }

    // Re-sort the dataset.  We have
    // [ used | far | other used ]
    // and we want
    // [ far | all used ].
    SortPointSet(indices, distances, 0, usedSetSize, farSetSize);

    return;
  }

  const int nextScale = std::min(scale - 1,
      (int) ceil(log(maxDistance) / log(expansionConstant)) - 1);
  const double bound = pow(expansionConstant, nextScale);

  // This needs to be taken out.  It's a sanity check for now.
  Log::Assert(nextScale < scale);

  // First, make the self child.  We must split the given near set into the near
  // set and far set for the self child.
  size_t childNearSetSize =
      SplitNearFar(indices, distances, bound, nearSetSize);

  // Build the self child (recursively).
  size_t childFarSetSize = nearSetSize - childNearSetSize;
  size_t childUsedSetSize = 0;
  children.push_back(new CoverTree(dataset, expansionConstant, pointIndex,
      nextScale, indices, distances, childNearSetSize, childFarSetSize,
      childUsedSetSize));

  // Now the arrays, in memory, look like this:
  // [ childFar | childUsed | far | used ]
  // but we need to move the used points past our far set:
  // [ childFar | far | childUsed + used ]
  // and keeping in mind that childFar = our near set,
  // [ near | far | childUsed + used ]
  // is what we are trying to make.
  SortPointSet(indices, distances, childFarSetSize, childUsedSetSize,
      farSetSize);

  // The self-child should not have used all the points in the near set.  If it
  // did, this is an implicit node.
  Log::Assert(childUsedSetSize < nearSetSize);

  // Update size of near set and used set.
  nearSetSize -= childUsedSetSize;
  usedSetSize += childUsedSetSize;

  // Now for each point in the near set, we need to make children.  To save
  // computation later, we'll create an array holding the points in the near
  // set, and then after each run we'll check which of those (if any) were used
  // and we will remove them.  ...if that's faster.  I think it is.
  arma::Col<size_t> nearSet = indices.rows(0, nearSetSize - 1);
  for (size_t i = 0; i < nearSet.n_elem; ++i)
  {
    // If this point has been used, skip to the next one.
    if (nearSet[i] == dataset.n_cols)
      continue;

    const size_t newPointIndex = nearSet[i]; // nearSet holds indices.

    // We need to move this point into the used set.  To do this we'll swap it
    // with the last value in the far set and then increment the counters
    // accordingly.  We don't have to worry about the fact that the point we
    // swapped is actually in the far set but grouped with the near set, because
    // we're about to rebuild that anyway.
    size_t setIndex;
    for (size_t k = 0; k < nearSetSize + farSetSize; ++k)
      if (indices[k] == newPointIndex)
        setIndex = k;

    // Ensure we need to swap.
    if (setIndex != ((nearSetSize + farSetSize) - 1))
    {
      // Perform the swap.
      const size_t otherLocation = (nearSetSize + farSetSize) - 1;
      const double tmpDist = distances[setIndex];

      indices[setIndex] = indices[otherLocation];
      distances[setIndex] = distances[otherLocation];

      indices[otherLocation] = newPointIndex;
      distances[otherLocation] = tmpDist;
    }

    // Update the near set size.  The used set size is updated by the recursive
    // child constructor (but we have to add one for the point we are using,
    // because the child constructor will not count that).
    nearSetSize--;
    usedSetSize++;

    // Rebuild the distances for this point.
    ComputeDistances(newPointIndex, indices, distances,
        nearSetSize + farSetSize);

    // Split into near and far sets for this point.
    childNearSetSize = SplitNearFar(indices, distances, bound, nearSetSize +
        farSetSize);

    // Build this child (recursively).
    childUsedSetSize = 0;
    childFarSetSize = ((nearSetSize + farSetSize) - childNearSetSize);
    children.push_back(new CoverTree(dataset, expansionConstant, newPointIndex,
        nextScale, indices, distances, childNearSetSize, childFarSetSize,
        childUsedSetSize));

    // Now the arrays, in memory, look like this:
    // [ childFar | childUsed | used ]
    // So we don't really need to do anything to them to get ready for the next
    // round.  We do need to look at the points that were used and update the
    // nearSet array.  This double for loop is suboptimal, but the best way I
    // can think of to do this.
    size_t usedNearSetPoints = 0;
    for (size_t j = childFarSetSize; j < (childFarSetSize + childUsedSetSize);
         ++j)
    {
      for (size_t k = i + 1; k < nearSet.n_elem; ++k)
      {
        if (indices[j] == nearSet[k])
        {
          nearSet[k] = dataset.n_cols; // Invalid index to indicate it's used.
          usedNearSetPoints++;
        }
      }
    }

    // Now we update the count of far set points and near set points.
    farSetSize -= (childUsedSetSize - usedNearSetPoints);
    nearSetSize -= usedNearSetPoints;

    // Update number of used points.
    usedSetSize += childUsedSetSize;
  }

  // Now that this is all done, our memory looks like this:
  // [ childFar | childUsed | used ]
  // We need to rebuild the distances and the set so it looks like this:
  // [ far | used ]
  // because all the points in the near set should be used up.
  farSetSize = childFarSetSize;

  // No need to rebuild the distances if we never modified them.
  if (nearSet.n_elem != 0)
    ComputeDistances(pointIndex, indices, distances, farSetSize);
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
CoverTree<MetricType, RootPointPolicy, StatisticType>::~CoverTree()
{
  // Delete each child.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
double CoverTree<MetricType, RootPointPolicy, StatisticType>::MinDistance(
    const CoverTree<MetricType, RootPointPolicy, StatisticType>* other) const
{
  // Every cover tree node will contain points up to EC^(scale + 1) away.
  return MetricType::Evaluate(dataset.col(point),
      other->Dataset().col(other->Point())) -
      std::pow(expansionConstant, scale + 1) -
      std::pow(other->ExpansionConstant(), other->Scale() + 1);
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
double CoverTree<MetricType, RootPointPolicy, StatisticType>::MinDistance(
    const arma::vec& other) const
{
  return MetricType::Evaluate(dataset.unsafe_col(point), other) -
      std::pow(expansionConstant, scale + 1);
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
double CoverTree<MetricType, RootPointPolicy, StatisticType>::MaxDistance(
    const CoverTree<MetricType, RootPointPolicy, StatisticType>* other) const
{
  return MetricType::Evaluate(dataset.col(point),
      other->Dataset().col(other->Point())) +
      std::pow(expansionConstant, scale + 1) +
      std::pow(other->ExpansionConstant(), other->Scale() + 1);
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
double CoverTree<MetricType, RootPointPolicy, StatisticType>::MaxDistance(
    const arma::vec& other) const
{
  return MetricType::Evaluate(dataset.unsafe_col(point), other) +
      std::pow(expansionConstant, scale + 1);
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
size_t CoverTree<MetricType, RootPointPolicy, StatisticType>::SplitNearFar(
    arma::Col<size_t>& indices,
    arma::vec& distances,
    const double bound,
    const size_t pointSetSize)
{
  // Sanity check; there is no guarantee that this condition will not be true.
  // ...or is there?
  if (pointSetSize <= 1)
    return 0;

  // We'll traverse from both left and right.
  size_t left = 0;
  size_t right = pointSetSize - 1;

  // A modification of quicksort, with the pivot value set to the bound.
  // Everything on the left of the pivot will be less than or equal to the
  // bound; everything on the right will be greater than the bound.
  while ((distances[left] <= bound) && (left != right))
    ++left;
  while ((distances[right] > bound) && (left != right))
    --right;

  while (left != right)
  {
    // Now swap the values and indices.
    const size_t tempPoint = indices[left];
    const double tempDist = distances[left];

    indices[left] = indices[right];
    distances[left] = distances[right];

    indices[right] = tempPoint;
    distances[right] = tempDist;

    // Traverse the left, seeing how many points are correctly on that side.
    // When we encounter an incorrect point, stop.  We will switch it later.
    while ((distances[left] <= bound) && (left != right))
      ++left;

    // Traverse the right, seeing how many points are correctly on that side.
    // When we encounter an incorrect point, stop.  We will switch it with the
    // wrong point from the left side.
    while ((distances[right] > bound) && (left != right))
      --right;
  }

  // The final left value is the index of the first far value.
  return left;
}

// Returns the maximum distance between points.
template<typename MetricType, typename RootPointPolicy, typename StatisticType>
void CoverTree<MetricType, RootPointPolicy, StatisticType>::ComputeDistances(
    const size_t pointIndex,
    const arma::Col<size_t>& indices,
    arma::vec& distances,
    const size_t pointSetSize)
{
  // For each point, rebuild the distances.  The indices do not need to be
  // modified.
  for (size_t i = 0; i < pointSetSize; ++i)
  {
    distances[i] = MetricType::Evaluate(dataset.col(pointIndex),
        dataset.col(indices[i]));
  }
}

template<typename MetricType, typename RootPointPolicy, typename StatisticType>
size_t CoverTree<MetricType, RootPointPolicy, StatisticType>::SortPointSet(
    arma::Col<size_t>& indices,
    arma::vec& distances,
    const size_t childFarSetSize,
    const size_t childUsedSetSize,
    const size_t farSetSize)
{
  // We'll use low-level memcpy calls ourselves, just to ensure it's done
  // quickly and the way we want it to be.  Unfortunately this takes up more
  // memory than one-element swaps, but there's not a great way around that.
  const size_t bufferSize = std::min(farSetSize, childUsedSetSize);
  const size_t bigCopySize = std::max(farSetSize, childUsedSetSize);

  // Sanity check: there is no need to sort if the buffer size is going to be
  // zero.
  if (bufferSize == 0)
    return (childFarSetSize + farSetSize);

  size_t* indicesBuffer = new size_t[bufferSize];
  double* distancesBuffer = new double[bufferSize];

  // The start of the memory region to copy to the buffer.
  const size_t bufferFromLocation = ((bufferSize == farSetSize) ?
      (childFarSetSize + childUsedSetSize) : childFarSetSize);
  // The start of the memory region to move directly to the new place.
  const size_t directFromLocation = ((bufferSize == farSetSize) ?
      childFarSetSize : (childFarSetSize + childUsedSetSize));
  // The destination to copy the buffer back to.
  const size_t bufferToLocation = ((bufferSize == farSetSize) ?
      childFarSetSize : (childFarSetSize + farSetSize));
  // The destination of the directly moved memory region.
  const size_t directToLocation = ((bufferSize == farSetSize) ?
      (childFarSetSize + farSetSize) : childFarSetSize);

  // Copy the smaller piece to the buffer.
  memcpy(indicesBuffer, indices.memptr() + bufferFromLocation,
      sizeof(size_t) * bufferSize);
  memcpy(distancesBuffer, distances.memptr() + bufferFromLocation,
      sizeof(double) * bufferSize);

  // Now move the other memory.
  memmove(indices.memptr() + directToLocation,
      indices.memptr() + directFromLocation, sizeof(size_t) * bigCopySize);
  memmove(distances.memptr() + directToLocation,
      distances.memptr() + directFromLocation, sizeof(double) * bigCopySize);

  // Now copy the temporary memory to the right place.
  memcpy(indices.memptr() + bufferToLocation, indicesBuffer,
      sizeof(size_t) * bufferSize);
  memcpy(distances.memptr() + bufferToLocation, distancesBuffer,
      sizeof(size_t) * bufferSize);

  delete[] indicesBuffer;
  delete[] distancesBuffer;

  // This returns the complete size of the far set.
  return (childFarSetSize + farSetSize);
}

}; // namespace tree
}; // namespace mlpack

#endif
