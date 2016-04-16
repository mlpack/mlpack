/**
 * @file cover_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of CoverTree class.
 */
#ifndef __MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_IMPL_HPP

// In case it hasn't already been included.
#include "cover_tree.hpp"

#include <mlpack/core/util/string_util.hpp>
#include <string>

namespace mlpack {
namespace tree {

// Create the cover tree.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    MetricType* metric) :
    dataset(new MatType(dataset)),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localMetric(metric == NULL),
    localDataset(false),
    metric(metric),
    distanceComps(0)
{
  // If we need to create a metric, do that.  We'll just do it on the heap.
  if (localMetric)
    this->metric = new MetricType();

  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset.n_cols <= 1)
    return;

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset.n_cols - 1, dataset.n_cols - 1);
  // This is now [1 2 3 4 ... n].  We now ensure that the initial root node is
  // 0. So we swap dataset points and set point to 0.
  if (point != 0) {
    this->dataset->swap_cols(0, point);
    point = 0;
  }

  // Now the distances will be of size dataset.n_cols as we are using the
  // same vector to perform all the distances computation.
  arma::vec distances(dataset.n_cols);

  // Build the initial distances.
  ComputeDistances(point, distances, dataset.n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  // The logics asks us to maintain all the used variables on the left
  // side of the array. Since the point has been taken the usedSetSize
  // is 1 now.
  size_t usedSetSize = 1;
  CreateChildren(indices, distances, dataset.n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly, and rebuild the statistic.
      old->Child(i).Parent() = this;
      old->Child(i).Stat() = StatisticType(old->Child(i));
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.
  scale = (int) ceil(log(furthestDescendantDistance) / log(base));

  // Initialize statistic.
  stat = StatisticType(*this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    MetricType& metric,
    const ElemType base) :
    dataset(new MatType(dataset)),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localMetric(false),
    localDataset(false),
    metric(&metric),
    distanceComps(0)
{
  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset.n_cols <= 1)
    return;

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset.n_cols - 1, dataset.n_cols - 1);
  // This is now [1 2 3 4 ... n].  We now ensure that the initial root node is
  // 0. So we swap dataset points and set point to 0.
  if (point != 0) {
    this->dataset->swap_cols(0, point);
    point = 0;
  }

  // Now the distances will be of size dataset.n_cols as we are using the
  // same vector to perform all the distances computation.
  arma::vec distances(dataset.n_cols);

  // Build the initial distances.
  ComputeDistances(point, distances, dataset.n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  // The logics asks us to maintain all the used variables on the left
  // side of the array. Since the point has been taken the usedSetSize
  // is 1 now.
  size_t usedSetSize = 1;
  CreateChildren(indices, distances, dataset.n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly.
      old->Child(i).Parent() = this;
      // Rebuild the statistic.
      old->Child(i).Stat() = StatisticType(old->Child(i));
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.
  scale = (int) ceil(log(furthestDescendantDistance) / log(base));

  // Initialize statistic.
  stat = StatisticType(*this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    MatType&& data,
    const ElemType base) :
    dataset(new MatType(std::move(data))),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localMetric(true),
    localDataset(true),
    distanceComps(0)
{
  // We need to create a metric.  We'll just do it on the heap.
  this->metric = new MetricType();

  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset->n_cols <= 1)
    return;

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset->n_cols - 1, dataset->n_cols - 1);
  // This is now [1 2 3 4 ... n].  We now ensure that the initial root node is
  // 0. So we swap dataset points and set point to 0.
  if (point != 0) {
    dataset->swap_cols(0, point);
    point = 0;
  }

  // Now the distances will be of size dataset.n_cols as we are using the
  // same vector to perform all the distances computation.
  arma::vec distances(dataset->n_cols);

  // Build the initial distances.
  ComputeDistances(point, distances, dataset->n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  // The logics asks us to maintain all the used variables on the left
  // side of the array. Since the point has been taken the usedSetSize
  // is 1 now.
  size_t usedSetSize = 1;
  CreateChildren(indices, distances, dataset->n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly, and rebuild the statistic.
      old->Child(i).Parent() = this;
      old->Child(i).Stat() = StatisticType(old->Child(i));
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.
  scale = (int) ceil(log(furthestDescendantDistance) / log(base));

  // Initialize statistic.
  stat = StatisticType(*this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    MatType&& data,
    MetricType& metric,
    const ElemType base) :
    dataset(new MatType(std::move(data))),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localMetric(false),
    localDataset(true),
    metric(&metric),
    distanceComps(0)
{
  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset->n_cols <= 1)
    return;

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset->n_cols - 1, dataset->n_cols - 1);
  // This is now [1 2 3 4 ... n].  We now ensure that the initial root node is
  // 0. So we swap dataset points and set point to 0.
  if (point != 0) {
    dataset->swap_cols(0, point);
    point = 0;
  }

  // Now the distances will be of size dataset.n_cols as we are using the
  // same vector to perform all the distances computation.
  arma::vec distances(dataset->n_cols);

  // Build the initial distances.
  ComputeDistances(point, distances, dataset->n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  // The logics asks us to maintain all the used variables on the left
  // side of the array. Since the point has been taken the usedSetSize
  // is 1 now.
  size_t usedSetSize = 1;
  CreateChildren(indices, distances, dataset->n_cols - 1, farSetSize,
      usedSetSize);

  // If we ended up creating only one child, remove the implicit node.
  while (children.size() == 1)
  {
    // Prepare to delete the implicit child node.
    CoverTree* old = children[0];

    // Now take its children and set their parent correctly.
    children.erase(children.begin());
    for (size_t i = 0; i < old->NumChildren(); ++i)
    {
      children.push_back(&(old->Child(i)));

      // Set its parent correctly, and rebuild the statistic.
      old->Child(i).Parent() = this;
      old->Child(i).Stat() = StatisticType(old->Child(i));
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.
  scale = (int) ceil(log(furthestDescendantDistance) / log(base));

  // Initialize statistic.
  stat = StatisticType(*this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    const size_t pointIndex,
    const int scale,
    CoverTree* parent,
    const ElemType parentDistance,
    arma::Col<size_t>& indices,
    arma::vec& distances,
    size_t nearSetSize,
    size_t& farSetSize,
    size_t& usedSetSize,
    MetricType& metric) :
    dataset(new MatType(dataset)),
    point(pointIndex),
    scale(scale),
    base(base),
    numDescendants(0),
    parent(parent),
    parentDistance(parentDistance),
    furthestDescendantDistance(0),
    localMetric(false),
    localDataset(false),
    metric(&metric),
    distanceComps(0)
{
  // If the size of the near set is 0, this is a leaf.
  if (nearSetSize == 0)
  {
    this->scale = INT_MIN;
    numDescendants = 1;
    stat = StatisticType(*this);
    return;
  }

  // Otherwise, create the children.
  CreateChildren(indices, distances, nearSetSize, farSetSize, usedSetSize);

  // Initialize statistic.
  stat = StatisticType(*this);
}

// Manually create a cover tree node.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    const size_t pointIndex,
    const int scale,
    CoverTree* parent,
    const ElemType parentDistance,
    const ElemType furthestDescendantDistance,
    MetricType* metric) :
    dataset(new MatType(dataset)),
    point(pointIndex),
    scale(scale),
    base(base),
    numDescendants(0),
    parent(parent),
    parentDistance(parentDistance),
    furthestDescendantDistance(furthestDescendantDistance),
    localMetric(metric == NULL),
    localDataset(false),
    metric(metric),
    distanceComps(0)
{
  // If necessary, create a local metric.
  if (localMetric)
    this->metric = new MetricType();

  // Initialize the statistic.
  stat = StatisticType(*this);
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const CoverTree& other) :
    dataset((other.parent == NULL) ? new MatType(*other.dataset) : NULL),
    point(other.point),
    scale(other.scale),
    base(other.base),
    stat(other.stat),
    numDescendants(other.numDescendants),
    parent(other.parent),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    localMetric(false),
    localDataset(other.parent == NULL),
    metric(other.metric),
    distanceComps(0)
{
  // Copy each child by hand.
  for (size_t i = 0; i < other.NumChildren(); ++i)
  {
    children.push_back(new CoverTree(other.Child(i)));
    children[i]->Parent() = this;
    children[i]->dataset = this->dataset;
  }
}

// Construct from a boost::serialization archive.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename Archive>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    Archive& ar,
    const typename boost::enable_if<typename Archive::is_loading>::type*) :
    CoverTree() // Create an empty CoverTree.
{
  // Now, serialize to our empty tree.
  ar >> data::CreateNVP(*this, "tree");
}


template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::~CoverTree()
{
  // Delete each child.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];

  // Delete the local metric, if necessary.
  if (localMetric)
    delete metric;

  // Delete the local dataset, if necessary.
  if (localDataset)
    delete dataset;
}

//! Return the number of descendant points.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline size_t
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    NumDescendants() const
{
  return numDescendants;
}

//! Return the index of a particular descendant point.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline size_t
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::Descendant(
    const size_t index) const
{
  // The first descendant is the point contained within this node.
  if (index == 0)
    return point;

  // Is it in the self-child?
  if (index < children[0]->NumDescendants())
    return children[0]->Descendant(index);

  // Now check the other children.
  size_t sum = children[0]->NumDescendants();
  for (size_t i = 1; i < children.size(); ++i)
  {
    if (index - sum < children[i]->NumDescendants())
      return children[i]->Descendant(index - sum);
    sum += children[i]->NumDescendants();
  }

  // This should never happen.
  return (size_t() - 1);
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const CoverTree* other) const
{
  // Every cover tree node will contain points up to base^(scale + 1) away.
  return std::max(metric->Evaluate(dataset->col(point),
      other->Dataset().col(other->Point())) -
      furthestDescendantDistance - other->FurthestDescendantDistance(), 0.0);
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const CoverTree* other, const ElemType distance) const
{
  // We already have the distance as evaluated by the metric.
  return std::max(distance - furthestDescendantDistance -
      other->FurthestDescendantDistance(), 0.0);
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const arma::vec& other) const
{
  return std::max(metric->Evaluate(dataset->col(point), other) -
      furthestDescendantDistance, 0.0);
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const arma::vec& /* other */, const ElemType distance) const
{
  return std::max(distance - furthestDescendantDistance, 0.0);
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const CoverTree* other) const
{
  return metric->Evaluate(dataset->col(point),
      other->Dataset().col(other->Point())) +
      furthestDescendantDistance + other->FurthestDescendantDistance();
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const CoverTree* other, const ElemType distance) const
{
  // We already have the distance as evaluated by the metric.
  return distance + furthestDescendantDistance +
      other->FurthestDescendantDistance();
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const arma::vec& other) const
{
  return metric->Evaluate(dataset->col(point), other) +
      furthestDescendantDistance;
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<MetricType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const arma::vec& /* other */, const ElemType distance) const
{
  return distance + furthestDescendantDistance;
}

//! Return the minimum and maximum distance to another node.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
math::RangeType<typename
    CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const CoverTree* other) const
{
  const ElemType distance = metric->Evaluate(dataset->col(point),
      other->Dataset().col(other->Point()));

  math::RangeType<ElemType> result;
  result.Lo() = distance - furthestDescendantDistance -
      other->FurthestDescendantDistance();
  result.Hi() = distance + furthestDescendantDistance +
      other->FurthestDescendantDistance();

  return result;
}

//! Return the minimum and maximum distance to another node given that the
//! point-to-point distance has already been calculated.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
math::RangeType<typename
    CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const CoverTree* other,
                  const ElemType distance) const
{
  math::RangeType<ElemType> result;
  result.Lo() = distance - furthestDescendantDistance -
      other->FurthestDescendantDistance();
  result.Hi() = distance + furthestDescendantDistance +
      other->FurthestDescendantDistance();

  return result;
}

//! Return the minimum and maximum distance to another point.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
math::RangeType<typename
    CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const arma::vec& other) const
{
  const ElemType distance = metric->Evaluate(dataset->col(point), other);

  return math::RangeType<ElemType>(distance - furthestDescendantDistance,
                     distance + furthestDescendantDistance);
}

//! Return the minimum and maximum distance to another point given that the
//! point-to-point distance has already been calculated.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
math::RangeType<typename
    CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const arma::vec& /* other */,
                  const ElemType distance) const
{
  return math::RangeType<ElemType>(distance - furthestDescendantDistance,
                     distance + furthestDescendantDistance);
}

//! For a newly initialized node, create children using the near and far set.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline void
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CreateChildren(
    arma::Col<size_t>& indices,
    arma::vec& distances,
    size_t nearSetSize,
    size_t& farSetSize,
    size_t& usedSetSize)
{
  // Determine the next scale level.  This should be the first level where there
  // are any points in the far set.  So, if we know the maximum distance in the
  // distances array, this will be the largest i such that
  //   maxDistance > pow(base, i)
  // and using this for the scale factor should guarantee we are not creating an
  // implicit node.  If the maximum distance is 0, every point in the near set
  // will be created as a leaf, and a child to this node.  We also do not need
  // to change the furthestChildDistance or furthestDescendantDistance.

  const ElemType maxDistance = max(distances.rows(usedSetSize,
      usedSetSize + nearSetSize + farSetSize - 1));
  const size_t storeUsedSetSize = usedSetSize;
  const size_t storeNearSetSize = nearSetSize;
  if (maxDistance == 0)
  {
    // Make the self child at the lowest possible level.
    // This should not modify farSetSize or usedSetSize.
    size_t tempSize = 0;
    children.push_back(new CoverTree(*dataset, base, point, INT_MIN, this, 0,
        indices, distances, 0, tempSize, usedSetSize, *metric));
    distanceComps += children.back()->DistanceComps();

    // Every point in the near set should be a leaf.
    size_t i = 0;
    while (i < nearSetSize)
    {
      // farSetSize and usedSetSize will not be modified.
      children.push_back(new CoverTree(*dataset, base, usedSetSize,
          INT_MIN, this, distances[usedSetSize], indices, distances, 0, tempSize,
          usedSetSize, *metric));
      distanceComps += children.back()->DistanceComps();
      usedSetSize++;
      i++;
    }

    // The number of descendants is just the number of children, because each of
    // them are leaves and contain one point.
    numDescendants = children.size();

    // Re-sort the dataset.  We have
    // [ used | far | other used ]
    // and we want
    // [ far | all used ].
    // I don't think there is any need of this anymore so commenting for time
    // being
    // SortPointSet(indices, distances, 0, usedSetSize, farSetSize);

    return;
  }
  const int nextScale = std::min(scale,
      (int) ceil(log(maxDistance) / log(base))) - 1;
  const ElemType bound = pow(base, nextScale);

  size_t changeNearSetSize = 0, changeFarSetSize = 0;
  // First, make the self child.  We must split the given near set into the near
  // set and far set for the self child.
  arma::Col<size_t> childIndices = arma::linspace<arma::Col<size_t> >(0,
    nearSetSize - 1, nearSetSize);
  //arma::vec childDistances(nearSetSize);
  size_t childNearSetSize =
      SplitNearFar(childIndices, distances, bound, usedSetSize, nearSetSize);

  // Build the self child (recursively).
  size_t childFarSetSize = nearSetSize - childNearSetSize;
  size_t childUsedSetSize = usedSetSize;
  
  children.push_back(new CoverTree(*dataset, base, point, nextScale, this, 0,
      indices, distances, childNearSetSize, childFarSetSize, childUsedSetSize,
      *metric));
  // Don't double-count the self-child (so, subtract one).
  numDescendants += children[0]->NumDescendants();

  // The self-child can't modify the furthestChildDistance away from 0, but it
  // can modify the furthestDescendantDistance.
  furthestDescendantDistance = children[0]->FurthestDescendantDistance();

  // Remove any implicit nodes we may have created.
  RemoveNewImplicitNodes();

  distanceComps += children[0]->DistanceComps();

  // Now the arrays, in memory, look like this:
  // [ childFar | childUsed | far | used ]
  // but we need to move the used points past our far set:
  // [ childFar | far | childUsed + used ]
  // and keeping in mind that childFar = our near set,
  // [ near | far | childUsed + used ]
  // is what we are trying to make.
  // Again commenting for later review.
  // SortPointSet(indices, distances, childFarSetSize, childUsedSetSize,
  //    farSetSize);

  // Update size of near set and used set.
  nearSetSize -= (childUsedSetSize - usedSetSize);
  usedSetSize += (childUsedSetSize - usedSetSize);

  // Now for each point in the near set, we need to make children.  To save
  // computation later, we'll create an array holding the points in the near
  // set, and then after each run we'll check which of those (if any) were used
  // and we will remove them.  ...if that's faster.  I think it is.
  while (nearSetSize > 0)
  {
    size_t newPointIndex = usedSetSize + nearSetSize - 1;

    // Swap to front if necessary. Here front is the position just after
    // the used set.
    if (newPointIndex != usedSetSize)
    {
      dataset->swap_cols(newPointIndex, usedSetSize);

      const ElemType tempDist = distances[newPointIndex];

      distances[newPointIndex] = distances[usedSetSize];

      distances[usedSetSize] = tempDist;

      newPointIndex = usedSetSize;
    }
    // Will this be a new furthest child?
    if (distances[newPointIndex] > furthestDescendantDistance)
      furthestDescendantDistance = distances[newPointIndex];

    // If there's only one point left, we don't need this crap.
    if ((nearSetSize == 1) && (farSetSize == 0))
    {
      size_t childNearSetSize = 0;

      children.push_back(new CoverTree(*dataset, base, newPointIndex, nextScale,
          this, distances[newPointIndex], indices, distances, childNearSetSize, farSetSize,
          usedSetSize, *metric));
      distanceComps += children.back()->DistanceComps();
      numDescendants += children.back()->NumDescendants();

      // Because the far set size is 0, we don't have to do any swapping to
      // move the point into the used set.
      ++usedSetSize;
      --nearSetSize;

      // And we're done.
      break;
    }

    // Create the near and far set indices and distance vectors.  We don't fill
    // in the self-point, yet.
    arma::Col<size_t> childIndices = arma::linspace<arma::Col<size_t> >(0,
      nearSetSize + farSetSize - 2, nearSetSize + farSetSize - 1);
    arma::vec childDistances(nearSetSize + farSetSize);

    // Build distances for the child.
    // But first make a copy of the existing distances
    size_t i;
    for (i = 0; i < nearSetSize + farSetSize; ++i)
    {
      childDistances[i] = distances[i + usedSetSize];
    }

    ComputeDistances(usedSetSize, distances,
        nearSetSize + farSetSize - 1);

    // Reuse the variables declared earlier.
    changeNearSetSize = 0;
    changeFarSetSize = 0;
    childUsedSetSize = usedSetSize + 1; // Mark self point as used.
    // Split into near and far sets for this point.
    childNearSetSize = SplitNearFar(childIndices, distances, bound,
        childUsedSetSize, nearSetSize + farSetSize - 1);
    childFarSetSize = PruneFarSet(childIndices, distances,
        base * bound, childNearSetSize, childUsedSetSize,
        (nearSetSize + farSetSize - 1));

    // Since the SplitNearFar and PruneFarSet change the dataset to the
    // form:
    // [used | childNear | childFar | other datapoints in near and far set]
    // We use the following function to update it to
    // [used | childNear | childFar | leftNear | leftFar | others]
    // We can think this as an subsitute to what happens in 
    // MoveToUsedSet function just the thing is we already maintain
    // the used set to left side, and instead want to change other points.
    UpdateDataset(childIndices, childDistances, distances, nearSetSize,
      farSetSize, childNearSetSize, childFarSetSize, usedSetSize,
      changeNearSetSize, changeFarSetSize);

    // Now that we know the near and far set sizes, we can put the used point
    // (the self point) in the correct place; now, when we call
    // MoveToUsedSet(), it will move the self-point correctly.  The distance
    // does not matter.
    // We are removing the use of indices and new distances vectors.
    // childIndices(childNearSetSize + childFarSetSize) = indices[0];
    // childDistances(childNearSetSize + childFarSetSize) = 0;

    // Build this child (recursively).
    
    children.push_back(new CoverTree(*dataset, base, usedSetSize, nextScale,
        this, distances[usedSetSize], childIndices, distances, childNearSetSize,
        childFarSetSize, childUsedSetSize, *metric));
    numDescendants += children.back()->NumDescendants();

    // Remove any implicit nodes.
    RemoveNewImplicitNodes();

    distanceComps += children.back()->DistanceComps();

    // Now with the child created, it returns the childIndices and
    // childDistances vectors in this form:
    // [ childFar | childUsed ]
    // For each point in the childUsed set, we must move that point to the used
    // set in our own vector.
    // Again since we are having changes, no need of this.
    // MoveToUsedSet(indices, distances, nearSetSize, farSetSize, usedSetSize,
    //    childIndices, childFarSetSize, childUsedSetSize);

    // Just update the used set and near set sizes
    nearSetSize = changeNearSetSize;
    farSetSize = changeFarSetSize;
    usedSetSize += (childUsedSetSize - usedSetSize);
  }
  // Calculate furthest descendant.
  for (size_t i = (storeUsedSetSize); i < (storeNearSetSize + storeUsedSetSize);
      ++i)
    if (distances[i] > furthestDescendantDistance)
      furthestDescendantDistance = distances[i];
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    UpdateDataset(arma::Col<size_t>& indices,
                 arma::vec tempDistances,
                 arma::vec& distances,
                 const size_t parNearSetSize,
                 const size_t parFarSetSize,
                 const size_t nearSetSize,
                 const size_t farSetSize,
                 const size_t usedSetSize,
                 size_t &changeNearSetSize,
                 size_t &changeFarSetSize)
{
  size_t i, j;
  std::vector<size_t> uncoveredIndices;
  for (i = 1; i < (parNearSetSize + parFarSetSize); i++)
  {
    for (j = 0; j < nearSetSize; j++)
      if(indices[j] + 1 == i)
        break;
    if (j == nearSetSize)
    {
      if(i < parNearSetSize)
      {
        uncoveredIndices.push_back(i);
        changeNearSetSize++;
      }
      if (i >= parNearSetSize && i < parFarSetSize)
        changeFarSetSize++;
    }

  }
  // Now Update the dataset
  // One major point is that the data set is properly arranged by
  // the SplitNearFar and PruneFarSet function. But we need that 
  // the dataset should look like
  // [used | near | far]
  // Currently it can be in any order, but the uncoveredIndices has
  // the indices that should be placed just after used.

  size_t replaceIndex = nearSetSize + 1;

  for (i = 0; i < uncoveredIndices.size(); i++)
  {
    for (j = nearSetSize; j < parNearSetSize +
      parFarSetSize; j++)
    {
      if (uncoveredIndices[i] == indices[j] + 1)
      {
        // Remember that the indices[j] is one behind because of 
        // the child was already included when calculating the 
        // near far sets.
        dataset->swap_cols(uncoveredIndices[i] + usedSetSize,
          replaceIndex + usedSetSize);
        size_t tempIndex = indices[j];
        indices[j] = indices[replaceIndex - 1];
        indices[replaceIndex - 1] = tempIndex;
        break;
      }
    }
    replaceIndex++;
  }

  // And we need to copy the rest of distances to original.
  for (i = nearSetSize + 1; i <= nearSetSize + farSetSize; i++)
  {
    distances[i + usedSetSize] = tempDistances[i - 1];
  }
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    SplitNearFar(arma::Col<size_t>& indices,
                 arma::vec& distances,
                 const ElemType bound,
                 const size_t usedSetSize,
                 const size_t pointSetSize)
{

  // Sanity check; there is no guarantee that this condition will not be true.
  // ...or is there?
  if (pointSetSize <= 1)
    return 0;

  // We'll traverse from both left and right.
  size_t left = usedSetSize, right = usedSetSize + pointSetSize - 1;

  // A modification of quicksort, with the pivot value set to the bound.
  // Everything on the left of the pivot will be less than or equal to the
  // bound; everything on the right will be greater than the bound.
  while ((distances[left] <= bound) && (left != right))
    ++left;
  while ((distances[right] > bound) && (left != right))
    --right;

  while (left != right)
  {
    // Earlier we were not swapping the dataset, but now we are swapping.
    dataset->swap_cols(left, right);

    // Now swap the distances only.
    const size_t tempPoint = indices[left - usedSetSize];
    const ElemType tempDist = distances[left];

    indices[left - usedSetSize] = indices[right - usedSetSize];
    distances[left] = distances[right];

    indices[right - usedSetSize] = tempPoint;
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
  return left - usedSetSize;
}

// Returns the maximum distance between points.
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    ComputeDistances(const size_t pointIndex,
                     arma::vec& distances,
                     const size_t pointSetSize)
{
  // For each point, rebuild the distances.  The indices do not need to be
  // modified.
  distanceComps += pointSetSize;
  for (size_t i = 0; i < pointSetSize; ++i)
  {
    distances[i + 1 + pointIndex] = metric->Evaluate(dataset->col(pointIndex),
        dataset->col(i + 1 + pointIndex));
  }
}


template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    MoveToUsedSet(arma::Col<size_t>& indices,
                  arma::vec& distances,
                  size_t& nearSetSize,
                  size_t& farSetSize,
                  size_t& usedSetSize,
                  arma::Col<size_t>& childIndices,
                  const size_t childFarSetSize, // childNearSetSize is 0 here.
                  const size_t childUsedSetSize)
{
  const size_t originalSum = nearSetSize + farSetSize + usedSetSize;

  // Loop across the set.  We will swap points as we need.  It should be noted
  // that farSetSize and nearSetSize may change with each iteration of this loop
  // (depending on if we make a swap or not).
  size_t startChildUsedSet = 0; // Where to start in the child set.
  for (size_t i = 0; i < nearSetSize; ++i)
  {
    // Discover if this point was in the child's used set.
    for (size_t j = startChildUsedSet; j < childUsedSetSize; ++j)
    {
      if (childIndices[childFarSetSize + j] == indices[i])
      {
        // We have found a point; a swap is necessary.

        // Since this point is from the near set, to preserve the near set, we
        // must do a swap.
        if (farSetSize > 0)
        {
          if ((nearSetSize - 1) != i)
          {
            // In this case it must be a three-way swap.
            size_t tempIndex = indices[nearSetSize + farSetSize - 1];
            ElemType tempDist = distances[nearSetSize + farSetSize - 1];

            size_t tempNearIndex = indices[nearSetSize - 1];
            ElemType tempNearDist = distances[nearSetSize - 1];

            indices[nearSetSize + farSetSize - 1] = indices[i];
            distances[nearSetSize + farSetSize - 1] = distances[i];

            indices[nearSetSize - 1] = tempIndex;
            distances[nearSetSize - 1] = tempDist;

            indices[i] = tempNearIndex;
            distances[i] = tempNearDist;
          }
          else
          {
            // We can do a two-way swap.
            size_t tempIndex = indices[nearSetSize + farSetSize - 1];
            ElemType tempDist = distances[nearSetSize + farSetSize - 1];

            indices[nearSetSize + farSetSize - 1] = indices[i];
            distances[nearSetSize + farSetSize - 1] = distances[i];

            indices[i] = tempIndex;
            distances[i] = tempDist;
          }
        }
        else if ((nearSetSize - 1) != i)
        {
          // A two-way swap is possible.
          size_t tempIndex = indices[nearSetSize + farSetSize - 1];
          ElemType tempDist = distances[nearSetSize + farSetSize - 1];

          indices[nearSetSize + farSetSize - 1] = indices[i];
          distances[nearSetSize + farSetSize - 1] = distances[i];

          indices[i] = tempIndex;
          distances[i] = tempDist;
        }
        else
        {
          // No swap is necessary.
        }

        // We don't need to do a complete preservation of the child index set,
        // but we want to make sure we only loop over points we haven't seen.
        // So increment the child counter by 1 and move a point if we need.
        if (j != startChildUsedSet)
        {
          childIndices[childFarSetSize + j] = childIndices[childFarSetSize +
              startChildUsedSet];
        }

        // Update all counters from the swaps we have done.
        ++startChildUsedSet;
        --nearSetSize;
        --i; // Since we moved a point out of the near set we must step back.

        break; // Break out of this for loop; back to the first one.
      }
    }
  }

  // Now loop over the far set.  This loop is different because we only require
  // a normal two-way swap instead of the three-way swap to preserve the near
  // set / far set ordering.
  for (size_t i = 0; i < farSetSize; ++i)
  {
    // Discover if this point was in the child's used set.
    for (size_t j = startChildUsedSet; j < childUsedSetSize; ++j)
    {
      if (childIndices[childFarSetSize + j] == indices[i + nearSetSize])
      {
        // We have found a point to swap.

        // Perform the swap.
        size_t tempIndex = indices[nearSetSize + farSetSize - 1];
        ElemType tempDist = distances[nearSetSize + farSetSize - 1];

        indices[nearSetSize + farSetSize - 1] = indices[nearSetSize + i];
        distances[nearSetSize + farSetSize - 1] = distances[nearSetSize + i];

        indices[nearSetSize + i] = tempIndex;
        distances[nearSetSize + i] = tempDist;

        if (j != startChildUsedSet)
        {
          childIndices[childFarSetSize + j] = childIndices[childFarSetSize +
              startChildUsedSet];
        }

        // Update all counters from the swaps we have done.
        ++startChildUsedSet;
        --farSetSize;
        --i;

        break; // Break out of this for loop; back to the first one.
      }
    }
  }

  // Update used set size.
  usedSetSize += childUsedSetSize;

  Log::Assert(originalSum == (nearSetSize + farSetSize + usedSetSize));
}

template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    PruneFarSet(arma::Col<size_t>& indices,
                arma::vec& distances,
                const ElemType bound,
                const size_t nearSetSize,
                const size_t usedSetSize,
                const size_t pointSetSize)
{

  // Now we are just trying to keep the far set just next to the close set.
  // This will just make sure that the rest of the datset points are kept
  // away from the set this node is dealing with.
  size_t left = usedSetSize + nearSetSize;
  size_t right = usedSetSize + pointSetSize - 1;
  while ((distances[left] <= bound) && (left != right))
    ++left;
  while ((distances[right] > bound) && (left != right))
    --right;

  while (left != right)
  {
    dataset->swap_cols(left, right);

    const size_t tempPoint = indices[left - usedSetSize];
    const ElemType tempDist = distances[left];

    indices[left - usedSetSize] = indices[right - usedSetSize];
    distances[left] = distances[right];

    indices[right - usedSetSize] = tempPoint;
    distances[right] = tempDist;

    // Advance to next location which needs to switch.
    while ((distances[left] <= bound) && (left != right))
      ++left;
    while ((distances[right] > bound) && (left != right))
      --right;
  }

  // The far set size is the left pointer, with the near set size subtracted
  // from it.
  return (left - nearSetSize - usedSetSize);
}

/**
 * Take a look at the last child (the most recently created one) and remove any
 * implicit nodes that have been created.
 */
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::
    RemoveNewImplicitNodes()
{
  // If we created an implicit node, take its self-child instead (this could
  // happen multiple times).
  while (children[children.size() - 1]->NumChildren() == 1)
  {
    CoverTree* old = children[children.size() - 1];
    children.erase(children.begin() + children.size() - 1);

    // Now take its child.
    children.push_back(&(old->Child(0)));

    // Set its parent and parameters correctly, and rebuild the statistic.
    old->Child(0).Parent() = this;
    old->Child(0).ParentDistance() = old->ParentDistance();
    old->Child(0).DistanceComps() = old->DistanceComps();
    old->Child(0).Stat() = StatisticType(old->Child(0));

    // Remove its child (so it doesn't delete it).
    old->Children().erase(old->Children().begin() + old->Children().size() - 1);

    // Now delete it.
    delete old;
  }
}

/**
 * Default constructor, only for use with boost::serialization.
 */
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::CoverTree() :
    dataset(NULL),
    point(0),
    scale(INT_MIN),
    base(0.0),
    numDescendants(0),
    parent(NULL),
    parentDistance(0.0),
    furthestDescendantDistance(0.0),
    localMetric(false),
    localDataset(false),
    metric(NULL)
{
  // Nothing to do.
}

/**
 * Serialize to/from a boost::serialization archive.
 */
template<
    typename MetricType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename Archive>
void CoverTree<MetricType, StatisticType, MatType, RootPointPolicy>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  using data::CreateNVP;

  // If we're loading, and we have children, they need to be deleted.  We may
  // also need to delete the local metric and dataset.
  if (Archive::is_loading::value)
  {
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];

    if (localMetric && metric)
      delete metric;
    if (localDataset && dataset)
      delete dataset;
  }

  ar & CreateNVP(dataset, "dataset");
  ar & CreateNVP(point, "point");
  ar & CreateNVP(scale, "scale");
  ar & CreateNVP(base, "base");
  ar & CreateNVP(stat, "stat");
  ar & CreateNVP(numDescendants, "numDescendants");

  // Due to quirks of boost::serialization, depending on how the user
  // serializes the tree, it's possible that the root of the tree will
  // accidentally be serialized twice.  So if we are a first-level child, we
  // avoid serializing the parent.  The true (non-duplicated) parent will fix
  // the parent link.
  if (Archive::is_saving::value && parent != NULL && parent->Parent() == NULL)
  {
    CoverTree* fakeParent = NULL;
    ar & CreateNVP(fakeParent, "parent");
  }
  else
  {
    ar & CreateNVP(parent, "parent");
  }

  ar & CreateNVP(parentDistance, "parentDistance");
  ar & CreateNVP(furthestDescendantDistance, "furthestDescendantDistance");
  ar & CreateNVP(metric, "metric");

  if (Archive::is_loading::value && parent == NULL)
  {
    localMetric = true;
    localDataset = true;
  }

  // Lastly, serialize the children.
  size_t numChildren = children.size();
  ar & CreateNVP(numChildren, "numChildren");
  if (Archive::is_loading::value)
    children.resize(numChildren);
  for (size_t i = 0; i < numChildren; ++i)
  {
    std::ostringstream oss;
    oss << "child" << i;
    ar & CreateNVP(children[i], oss.str());
  }

  if (Archive::is_loading::value && parent == NULL)
  {
    // Look through each child individually.
    for (size_t i = 0; i < children.size(); ++i)
    {
      children[i]->localMetric = false;
      children[i]->localDataset = false;
      children[i]->Parent() = this;
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif
