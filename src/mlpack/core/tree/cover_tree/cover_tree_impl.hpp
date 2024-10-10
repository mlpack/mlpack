/**
 * @file core/tree/cover_tree/cover_tree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of CoverTree class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_COVER_TREE_COVER_TREE_IMPL_HPP

// In case it hasn't already been included.
#include "cover_tree.hpp"

#include <queue>
#include <string>

#include <mlpack/core/util/log.hpp>

namespace mlpack {

// Build the statistics, bottom-up.
template<typename TreeType, typename StatisticType>
void BuildStatistics(TreeType* node)
{
  // Recurse first.
  for (size_t i = 0; i < node->NumChildren(); ++i)
    BuildStatistics<TreeType, StatisticType>(&node->Child(i));

  // Now build the statistic.
  node->Stat() = StatisticType(*node);
}

// Create the cover tree.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    DistanceType* distance) :
    dataset(&dataset),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localDistance(distance == NULL),
    localDataset(false),
    distance(distance),
    distanceComps(0)
{
  // If we need to create a distance metric, do that.  We'll just do it on the
  // heap.
  if (localDistance)
    this->distance = new DistanceType();

  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset.n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

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

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
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
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset.n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    DistanceType& distance,
    const ElemType base) :
    dataset(&dataset),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localDistance(true),
    localDataset(false),
    distance(new DistanceType(distance)),
    distanceComps(0)
{
  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset.n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

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

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
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
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset.n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
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
    localDistance(true),
    localDataset(true),
    distanceComps(0)
{
  // We need to create a distance metric.  We'll just do it on the heap.
  this->distance = new DistanceType();

  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset->n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset->n_cols - 1, dataset->n_cols - 1);
  // This is now [1 2 3 4 ... n].  We must be sure that our point does not
  // occur.
  if (point != 0)
    indices[point - 1] = 0; // Put 0 back into the set; remove what was there.

  arma::vec distances(dataset->n_cols - 1);

  // Build the initial distances.
  ComputeDistances(point, indices, distances, dataset->n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
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

      // Set its parent correctly.
      old->Child(i).Parent() = this;
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset->n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    MatType&& data,
    DistanceType& distance,
    const ElemType base) :
    dataset(new MatType(std::move(data))),
    point(RootPointPolicy::ChooseRoot(dataset)),
    scale(INT_MAX),
    base(base),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
    localDistance(true),
    localDataset(true),
    distance(new DistanceType(distance)),
    distanceComps(0)
{
  // If there is only one point or zero points in the dataset... uh, we're done.
  // Technically, if the dataset has zero points, our node is not correct...
  if (dataset->n_cols <= 1)
  {
    scale = INT_MIN;
    return;
  }

  // Kick off the building.  Create the indices array and the distances array.
  arma::Col<size_t> indices = arma::linspace<arma::Col<size_t> >(1,
      dataset->n_cols - 1, dataset->n_cols - 1);
  // This is now [1 2 3 4 ... n].  We must be sure that our point does not
  // occur.
  if (point != 0)
    indices[point - 1] = 0; // Put 0 back into the set; remove what was there.

  arma::vec distances(dataset->n_cols - 1);

  // Build the initial distances.
  ComputeDistances(point, indices, distances, dataset->n_cols - 1);

  // Create the children.
  size_t farSetSize = 0;
  size_t usedSetSize = 0;
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

      // Set its parent correctly.
      old->Child(i).Parent() = this;
    }

    // Remove all the children so they don't get erased.
    old->Children().clear();

    // Reduce our own scale.
    scale = old->Scale();

    // Now delete it.
    delete old;
  }

  // Use the furthest descendant distance to determine the scale of the root
  // node.  Note that if the root is a leaf, we can have scale INT_MIN, but if
  // it *isn't* a leaf, we need to mark the scale as one higher than INT_MIN, so
  // that the recursions don't fail.
  if (furthestDescendantDistance == 0.0 && dataset->n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0.0)
    scale = INT_MIN + 1;
  else
    scale = (int) std::ceil(std::log(furthestDescendantDistance) /
        std::log(base));

  // Initialize statistics recursively after the entire tree construction is
  // complete.
  BuildStatistics<CoverTree, StatisticType>(this);

  Log::Info << distanceComps << " distance computations during tree "
      << "construction." << std::endl;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
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
    DistanceType& distance) :
    dataset(&dataset),
    point(pointIndex),
    scale(scale),
    base(base),
    numDescendants(0),
    parent(parent),
    parentDistance(parentDistance),
    furthestDescendantDistance(0),
    localDistance(false),
    localDataset(false),
    distance(&distance),
    distanceComps(0)
{
  // If the size of the near set is 0, this is a leaf.
  if (nearSetSize == 0)
  {
    this->scale = INT_MIN;
    numDescendants = 1;
    return;
  }

  // Otherwise, create the children.
  CreateChildren(indices, distances, nearSetSize, farSetSize, usedSetSize);
}

// Manually create a cover tree node.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const MatType& dataset,
    const ElemType base,
    const size_t pointIndex,
    const int scale,
    CoverTree* parent,
    const ElemType parentDistance,
    const ElemType furthestDescendantDistance,
    DistanceType* distance) :
    dataset(&dataset),
    point(pointIndex),
    scale(scale),
    base(base),
    numDescendants(0),
    parent(parent),
    parentDistance(parentDistance),
    furthestDescendantDistance(furthestDescendantDistance),
    localDistance(distance == NULL),
    localDataset(false),
    distance(distance),
    distanceComps(0)
{
  // If necessary, create a local distance metric.
  if (localDistance)
    this->distance = new DistanceType();
}

// Copy Constructor.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    const CoverTree& other) :
    dataset((other.parent == NULL && other.localDataset) ?
        new MatType(*other.dataset) : other.dataset),
    point(other.point),
    scale(other.scale),
    base(other.base),
    stat(other.stat),
    numDescendants(other.numDescendants),
    parent(other.parent),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    localDistance(other.localDistance),
    localDataset(other.parent == NULL && other.localDataset),
    distance((other.localDistance ? new DistanceType() : other.distance)),
    distanceComps(0)
{
  // Copy each child by hand.
  for (size_t i = 0; i < other.NumChildren(); ++i)
  {
    children.push_back(new CoverTree(other.Child(i)));
    children[i]->Parent() = this;
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL && localDataset)
  {
    std::queue<CoverTree*> queue;

    for (size_t i = 0; i < NumChildren(); ++i)
      queue.push(children[i]);

    while (!queue.empty())
    {
      CoverTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      for (size_t i = 0; i < node->NumChildren(); ++i)
        queue.push(node->children[i]);
    }
  }
}

// Copy assignment operator: copy the given other tree.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>&
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
operator=(const CoverTree& other)
{
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  if (localDataset)
    delete dataset;

  if (localDistance)
    delete distance;

  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  dataset = ((other.parent == NULL && other.localDataset) ?
      new MatType(*other.dataset) : other.dataset);
  point = other.point;
  scale = other.scale;
  base = other.base;
  stat = other.stat;
  numDescendants = other.numDescendants;
  parent = other.parent;
  parentDistance = other.parentDistance;
  furthestDescendantDistance = other.furthestDescendantDistance;
  localDistance = other.localDistance;
  localDataset = (other.parent == NULL && other.localDataset);
  distance = (other.localDistance ? new DistanceType() : other.distance);
  distanceComps = 0;

  // Copy each child by hand.
  for (size_t i = 0; i < other.NumChildren(); ++i)
  {
    children.push_back(new CoverTree(other.Child(i)));
    children[i]->Parent() = this;
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL && localDataset)
  {
    std::queue<CoverTree*> queue;

    for (size_t i = 0; i < NumChildren(); ++i)
      queue.push(children[i]);

    while (!queue.empty())
    {
      CoverTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      for (size_t i = 0; i < node->NumChildren(); ++i)
        queue.push(node->children[i]);
    }
  }

  return *this;
}

// Move Constructor.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    CoverTree&& other) :
    dataset(other.dataset),
    point(other.point),
    children(std::move(other.children)),
    scale(other.scale),
    base(other.base),
    stat(std::move(other.stat)),
    numDescendants(other.numDescendants),
    parent(other.parent),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    localDistance(other.localDistance),
    localDataset(other.localDataset),
    distance(other.distance),
    distanceComps(other.distanceComps)
{
  // Set proper parent pointer.
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->Parent() = this;

  other.dataset = NULL;
  other.point = 0;
  other.scale = INT_MIN;
  other.base = 0;
  other.numDescendants = 0;
  other.parent = NULL;
  other.parentDistance = 0;
  other.furthestDescendantDistance = 0;
  other.localDistance = false;
  other.localDataset = false;
  other.distance = NULL;
}

// Move assignment operator: take ownership of the given tree.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>&
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
operator=(CoverTree&& other)
{
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  if (localDataset)
    delete dataset;

  if (localDistance)
    delete distance;

  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];

  dataset = other.dataset;
  point = other.point;
  children = std::move(other.children);
  scale = other.scale;
  base = other.base;
  stat = std::move(other.stat);
  numDescendants = other.numDescendants;
  parent = other.parent;
  parentDistance = other.parentDistance;
  furthestDescendantDistance = other.furthestDescendantDistance;
  localDistance = other.localDistance;
  localDataset = other.localDataset;
  distance = other.distance;
  distanceComps = other.distanceComps;

  // Set proper parent pointer.
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->Parent() = this;

  other.dataset = NULL;
  other.point = 0;
  other.scale = INT_MIN;
  other.base = 0;
  other.numDescendants = 0;
  other.parent = NULL;
  other.parentDistance = 0;
  other.furthestDescendantDistance = 0;
  other.localDistance = false;
  other.localDataset = false;
  other.distance = NULL;

  return *this;
}

// Construct from a cereal archive.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename Archive>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree(
    Archive& ar,
    const typename std::enable_if_t<cereal::is_loading<Archive>()>*) :
    CoverTree() // Create an empty CoverTree.
{
  // Now, serialize to our empty tree.
  ar(cereal::make_nvp("this", *this));
}


template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::~CoverTree()
{
  // Delete each child.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];

  // Delete the local distance metric, if necessary.
  if (localDistance)
    delete distance;

  // Delete the local dataset, if necessary.
  if (localDataset)
    delete dataset;
}

//! Return the number of descendant points.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline size_t
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    NumDescendants() const
{
  return numDescendants;
}

//! Return the index of a particular descendant point.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline size_t
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::Descendant(
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

/**
 * Return the index of the nearest child node to the given query point.  If
 * this is a leaf node, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
template<typename VecType>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetNearestChild(const VecType& point,
                    typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MinDistance(point);
    if (distance <= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Return the index of the furthest child node to the given query point.  If
 * this is a leaf node, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
template<typename VecType>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetFurthestChild(const VecType& point,
                     typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = 0;
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MaxDistance(point);
    if (distance >= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Return the index of the nearest child node to the given query node.  If it
 * can't decide, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetNearestChild(const CoverTree& queryNode)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MinDistance(queryNode);
    if (distance <= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Return the index of the furthest child node to the given query node.  If it
 * can't decide, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename RootPointPolicy>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    GetFurthestChild(const CoverTree& queryNode)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = 0;
  size_t bestIndex = 0;
  for (size_t i = 0; i < children.size(); ++i)
  {
    ElemType distance = children[i]->MaxDistance(queryNode);
    if (distance >= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const CoverTree& other) const
{
  // Every cover tree node will contain points up to base^(scale + 1) away.
  return std::max(distance->Evaluate(dataset->col(point),
      other.Dataset().col(other.Point())) -
      furthestDescendantDistance - other.FurthestDescendantDistance(), 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const CoverTree& other, const ElemType distance) const
{
  // We already have the distance as evaluated by the metric.
  return std::max(distance - furthestDescendantDistance -
      other.FurthestDescendantDistance(), 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const arma::vec& other) const
{
  return std::max(distance->Evaluate(dataset->col(point), other) -
      furthestDescendantDistance, 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MinDistance(const arma::vec& /* other */, const ElemType distance) const
{
  return std::max(distance - furthestDescendantDistance, 0.0);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const CoverTree& other) const
{
  return distance->Evaluate(dataset->col(point),
      other.Dataset().col(other.Point())) +
      furthestDescendantDistance + other.FurthestDescendantDistance();
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const CoverTree& other, const ElemType distance) const
{
  // We already have the distance as evaluated by the metric.
  return distance + furthestDescendantDistance +
      other.FurthestDescendantDistance();
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const arma::vec& other) const
{
  return distance->Evaluate(dataset->col(point), other) +
      furthestDescendantDistance;
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    MaxDistance(const arma::vec& /* other */, const ElemType distance) const
{
  return distance + furthestDescendantDistance;
}

//! Return the minimum and maximum distance to another node.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const CoverTree& other) const
{
  const ElemType dist = distance->Evaluate(dataset->col(point),
      other.Dataset().col(other.Point()));

  RangeType<ElemType> result;
  result.Lo() = std::max(dist - furthestDescendantDistance -
      other.FurthestDescendantDistance(), 0.0);
  result.Hi() = dist + furthestDescendantDistance +
      other.FurthestDescendantDistance();

  return result;
}

//! Return the minimum and maximum distance to another node given that the
//! point-to-point distance has already been calculated.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const CoverTree& other,
                  const ElemType distance) const
{
  RangeType<ElemType> result;
  result.Lo() = std::max(distance - furthestDescendantDistance -
      other.FurthestDescendantDistance(), 0.0);
  result.Hi() = distance + furthestDescendantDistance +
      other.FurthestDescendantDistance();

  return result;
}

//! Return the minimum and maximum distance to another point.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const arma::vec& other) const
{
  const ElemType dist = distance->Evaluate(dataset->col(point), other);

  return RangeType<ElemType>(
      std::max(dist - furthestDescendantDistance, 0.0),
      dist + furthestDescendantDistance);
}

//! Return the minimum and maximum distance to another point given that the
//! point-to-point distance has already been calculated.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    RangeDistance(const arma::vec& /* other */,
                  const ElemType distance) const
{
  return RangeType<ElemType>(
      std::max(distance - furthestDescendantDistance, 0.0),
      distance + furthestDescendantDistance);
}

//! For a newly initialized node, create children using the near and far set.
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline void
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    CreateChildren(arma::Col<size_t>& indices,
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
  const ElemType maxDistance = max(distances.rows(0,
      nearSetSize + farSetSize - 1));
  if (maxDistance == 0)
  {
    // Make the self child at the lowest possible level.
    // This should not modify farSetSize or usedSetSize.
    size_t tempSize = 0;
    children.push_back(new CoverTree(*dataset, base, point, INT_MIN, this, 0,
        indices, distances, 0, tempSize, usedSetSize, *distance));
    distanceComps += children.back()->DistanceComps();

    // Every point in the near set should be a leaf.
    for (size_t i = 0; i < nearSetSize; ++i)
    {
      // farSetSize and usedSetSize will not be modified.
      children.push_back(new CoverTree(*dataset, base, indices[i],
          INT_MIN, this, distances[i], indices, distances, 0, tempSize,
          usedSetSize, *distance));
      distanceComps += children.back()->DistanceComps();
      usedSetSize++;
    }

    // The number of descendants is just the number of children, because each of
    // them are leaves and contain one point.
    numDescendants = children.size();

    // Re-sort the dataset.  We have
    // [ used | far | other used ]
    // and we want
    // [ far | all used ].
    SortPointSet(indices, distances, 0, usedSetSize, farSetSize);

    return;
  }

  const int nextScale = std::min(scale,
      (int) std::ceil(std::log(maxDistance) / std::log(base))) - 1;
  const ElemType bound = std::pow(base, nextScale);

  // First, make the self child.  We must split the given near set into the near
  // set and far set for the self child.
  size_t childNearSetSize =
      SplitNearFar(indices, distances, bound, nearSetSize);

  // Build the self child (recursively).
  size_t childFarSetSize = nearSetSize - childNearSetSize;
  size_t childUsedSetSize = 0;
  children.push_back(new CoverTree(*dataset, base, point, nextScale, this, 0,
      indices, distances, childNearSetSize, childFarSetSize, childUsedSetSize,
      *distance));
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
  SortPointSet(indices, distances, childFarSetSize, childUsedSetSize,
      farSetSize);

  // Update size of near set and used set.
  nearSetSize -= childUsedSetSize;
  usedSetSize += childUsedSetSize;

  // Now for each point in the near set, we need to make children.  To save
  // computation later, we'll create an array holding the points in the near
  // set, and then after each run we'll check which of those (if any) were used
  // and we will remove them.  ...if that's faster.  I think it is.
  while (nearSetSize > 0)
  {
    size_t newPointIndex = nearSetSize - 1;

    // Swap to front if necessary.
    if (newPointIndex != 0)
    {
      const size_t tempIndex = indices[newPointIndex];
      const ElemType tempDist = distances[newPointIndex];

      indices[newPointIndex] = indices[0];
      distances[newPointIndex] = distances[0];

      indices[0] = tempIndex;
      distances[0] = tempDist;
    }

    // Will this be a new furthest child?
    if (distances[0] > furthestDescendantDistance)
      furthestDescendantDistance = distances[0];

    // If there's only one point left, we don't need this crap.
    if ((nearSetSize == 1) && (farSetSize == 0))
    {
      size_t childNearSetSize = 0;
      children.push_back(new CoverTree(*dataset, base, indices[0], nextScale,
          this, distances[0], indices, distances, childNearSetSize, farSetSize,
          usedSetSize, *distance));
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
    arma::Col<size_t> childIndices(nearSetSize + farSetSize);
    childIndices.rows(0, (nearSetSize + farSetSize - 2)) = indices.rows(1,
        nearSetSize + farSetSize - 1);
    arma::vec childDistances(nearSetSize + farSetSize);

    // Build distances for the child.
    ComputeDistances(indices[0], childIndices, childDistances, nearSetSize
        + farSetSize - 1);

    // Split into near and far sets for this point.
    childNearSetSize = SplitNearFar(childIndices, childDistances, bound,
        nearSetSize + farSetSize - 1);
    childFarSetSize = PruneFarSet(childIndices, childDistances,
        base * bound, childNearSetSize,
        (nearSetSize + farSetSize - 1));

    // Now that we know the near and far set sizes, we can put the used point
    // (the self point) in the correct place; now, when we call
    // MoveToUsedSet(), it will move the self-point correctly.  The distance
    // does not matter.
    childIndices(childNearSetSize + childFarSetSize) = indices[0];
    childDistances(childNearSetSize + childFarSetSize) = 0;

    // Build this child (recursively).
    childUsedSetSize = 1; // Mark self point as used.
    children.push_back(new CoverTree(*dataset, base, indices[0], nextScale,
        this, distances[0], childIndices, childDistances, childNearSetSize,
        childFarSetSize, childUsedSetSize, *distance));
    numDescendants += children.back()->NumDescendants();

    // Remove any implicit nodes.
    RemoveNewImplicitNodes();

    distanceComps += children.back()->DistanceComps();

    // Now with the child created, it returns the childIndices and
    // childDistances vectors in this form:
    // [ childFar | childUsed ]
    // For each point in the childUsed set, we must move that point to the used
    // set in our own vector.
    MoveToUsedSet(indices, distances, nearSetSize, farSetSize, usedSetSize,
        childIndices, childFarSetSize, childUsedSetSize);
  }

  // Calculate furthest descendant.
  for (size_t i = (nearSetSize + farSetSize); i < (nearSetSize + farSetSize +
      usedSetSize); ++i)
    if (distances[i] > furthestDescendantDistance)
      furthestDescendantDistance = distances[i];
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    SplitNearFar(arma::Col<size_t>& indices,
                 arma::vec& distances,
                 const ElemType bound,
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
    const ElemType tempDist = distances[left];

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
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    ComputeDistances(const size_t pointIndex,
                     const arma::Col<size_t>& indices,
                     arma::vec& distances,
                     const size_t pointSetSize)
{
  // For each point, rebuild the distances.  The indices do not need to be
  // modified.
  distanceComps += pointSetSize;
  for (size_t i = 0; i < pointSetSize; ++i)
  {
    distances[i] = distance->Evaluate(dataset->col(pointIndex),
        dataset->col(indices[i]));
  }
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    SortPointSet(arma::Col<size_t>& indices,
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
  ElemType* distancesBuffer = new ElemType[bufferSize];

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
      sizeof(ElemType) * bufferSize);

  // Now move the other memory.
  memmove(indices.memptr() + directToLocation,
      indices.memptr() + directFromLocation, sizeof(size_t) * bigCopySize);
  memmove(distances.memptr() + directToLocation,
      distances.memptr() + directFromLocation, sizeof(ElemType) * bigCopySize);

  // Now copy the temporary memory to the right place.
  memcpy(indices.memptr() + bufferToLocation, indicesBuffer,
      sizeof(size_t) * bufferSize);
  memcpy(distances.memptr() + bufferToLocation, distancesBuffer,
      sizeof(ElemType) * bufferSize);

  delete[] indicesBuffer;
  delete[] distancesBuffer;

  // This returns the complete size of the far set.
  return (childFarSetSize + farSetSize);
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
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
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
size_t CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
    PruneFarSet(arma::Col<size_t>& indices,
                arma::vec& distances,
                const ElemType bound,
                const size_t nearSetSize,
                const size_t pointSetSize)
{
  // What we are trying to do is remove any points greater than the bound from
  // the far set.  We don't care what happens to those indices and distances...
  // so, we don't need to properly swap points -- just drop new ones in place.
  size_t left = nearSetSize;
  size_t right = pointSetSize - 1;
  while ((distances[left] <= bound) && (left != right))
    ++left;
  while ((distances[right] > bound) && (left != right))
    --right;

  while (left != right)
  {
    // We don't care what happens to the point which should be on the right.
    indices[left] = indices[right];
    distances[left] = distances[right];
    --right; // Since we aren't changing the right.

    // Advance to next location which needs to switch.
    while ((distances[left] <= bound) && (left != right))
      ++left;
    while ((distances[right] > bound) && (left != right))
      --right;
  }

  // The far set size is the left pointer, with the near set size subtracted
  // from it.
  return (left - nearSetSize);
}

/**
 * Take a look at the last child (the most recently created one) and remove any
 * implicit nodes that have been created.
 */
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
inline void CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::
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

    // Set its parent and parameters correctly.
    old->Child(0).Parent() = this;
    old->Child(0).ParentDistance() = old->ParentDistance();
    old->Child(0).DistanceComps() = old->DistanceComps();

    // Remove its child (so it doesn't delete it).
    old->Children().erase(old->Children().begin() + old->Children().size() - 1);

    // Now delete it.
    delete old;
  }
}

/**
 * Default constructor, only for use with cereal.
 */
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::CoverTree() :
    dataset(NULL),
    point(0),
    scale(INT_MIN),
    base(0.0),
    numDescendants(0),
    parent(NULL),
    parentDistance(0.0),
    furthestDescendantDistance(0.0),
    localDistance(false),
    localDataset(false),
    distance(NULL),
    distanceComps(0)
{
  // Nothing to do.
}

/**
 * Serialize to/from a cereal archive.
 */
template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename Archive>
void
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  // If we're loading, and we have children, they need to be deleted.  We may
  // also need to delete the local distance metric and dataset.
  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];

    if (localDistance && distance)
      delete distance;
    if (localDataset && dataset)
      delete dataset;

    parent = NULL;
  }

  bool hasParent = (parent != NULL);
  ar(CEREAL_NVP(hasParent));
  MatType*& datasetTemp = const_cast<MatType*&>(dataset);
  if (!hasParent)
    ar(CEREAL_POINTER(datasetTemp));

  ar(CEREAL_NVP(point));
  ar(CEREAL_NVP(scale));
  ar(CEREAL_NVP(base));
  ar(CEREAL_NVP(stat));
  ar(CEREAL_NVP(numDescendants));
  ar(CEREAL_NVP(parentDistance));
  ar(CEREAL_NVP(furthestDescendantDistance));
  ar(CEREAL_POINTER(distance));

  if (cereal::is_loading<Archive>() && !hasParent)
  {
    localDistance = true;
    localDataset = true;
  }

  // Lastly, serialize the children.
  ar(CEREAL_VECTOR_POINTER(children));

  if (cereal::is_loading<Archive>())
  {
    // Look through each child individually.
    for (size_t i = 0; i < children.size(); ++i)
    {
      children[i]->localDistance = false;
      children[i]->localDataset = false;
      children[i]->Parent() = this;
    }
  }

  if (!hasParent)
  {
    std::stack<CoverTree*> stack;
    for (size_t i = 0; i < children.size(); ++i)
    {
      stack.push(children[i]);
    }
    while (!stack.empty())
    {
      CoverTree* node = stack.top();
      stack.pop();
      node->dataset = dataset;
      for (size_t i = 0; i < node->children.size(); ++i)
      {
        stack.push(node->children[i]);
      }
    }
  }
}

} // namespace mlpack

#endif
