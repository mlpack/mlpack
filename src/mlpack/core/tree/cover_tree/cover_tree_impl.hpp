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

  arma::vec distances(dataset.n_cols - 1, arma::fill::none);
  std::vector<bool> used(dataset.n_cols, false);

  // Build the initial distances.
  ComputeDistances(point, indices, used, distances);

  // Create the children.
  CreateChildren(indices, distances, used);

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
  if (furthestDescendantDistance == 0 && dataset.n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0)
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

  arma::vec distances(dataset.n_cols - 1, arma::fill::none);
  std::vector<bool> used(dataset.n_cols, false);

  // Build the initial distances.
  ComputeDistances(point, indices, used, distances);

  // Create the children.
  CreateChildren(indices, distances, used);

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
  if (furthestDescendantDistance == 0 && dataset.n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0)
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

  arma::vec distances(dataset->n_cols - 1, arma::fill::none);
  std::vector<bool> used(dataset->n_cols, false);

  // Build the initial distances.
  ComputeDistances(point, indices, used, distances);

  // Create the children.
  CreateChildren(indices, distances, used);

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
  if (furthestDescendantDistance == 0 && dataset->n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0)
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

  arma::vec distances(dataset->n_cols - 1, arma::fill::none);
  std::vector<bool> used(dataset->n_cols, false);

  // Build the initial distances.
  ComputeDistances(point, indices, used, distances);

  // Create the children.
  CreateChildren(indices, distances, used);

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
  if (furthestDescendantDistance == 0 && dataset->n_cols == 1)
    scale = INT_MIN;
  else if (furthestDescendantDistance == 0)
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
    std::vector<bool>& used,
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
  // Otherwise, create the children.
  CreateChildren(indices, distances, used);
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
      furthestDescendantDistance - other.FurthestDescendantDistance(),
      ElemType(0));
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
      other.FurthestDescendantDistance(), ElemType(0));
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename VecType>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::MinDistance(
    const VecType& other,
    const typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  return std::max(distance->Evaluate(dataset->col(point), other) -
      furthestDescendantDistance, ElemType(0));
}

template<
    typename DistanceType,
    typename StatisticType,
    typename MatType,
    typename RootPointPolicy
>
template<typename VecType>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::MinDistance(
    const VecType& /* other */,
    const ElemType distance,
    const typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  return std::max(distance - furthestDescendantDistance, ElemType(0));
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
template<typename VecType>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::MaxDistance(
    const VecType& other,
    const typename std::enable_if_t<IsVector<VecType>::value>*) const
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
template<typename VecType>
typename CoverTree<DistanceType, StatisticType, MatType,
    RootPointPolicy>::ElemType
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::MaxDistance(
    const VecType& /* other */,
    const ElemType distance,
    const typename std::enable_if_t<IsVector<VecType>::value>*) const
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
      other.FurthestDescendantDistance(), ElemType(0));
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
      other.FurthestDescendantDistance(), ElemType(0));
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
template<typename VecType>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::RangeDistance(
    const VecType& other,
    const typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  const ElemType dist = distance->Evaluate(dataset->col(point), other);

  return RangeType<ElemType>(
      std::max(dist - furthestDescendantDistance, ElemType(0)),
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
template<typename VecType>
RangeType<typename
    CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::ElemType>
CoverTree<DistanceType, StatisticType, MatType, RootPointPolicy>::RangeDistance(
    const VecType& /* other */,
    const ElemType distance,
    const typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  return RangeType<ElemType>(
      std::max(distance - furthestDescendantDistance, ElemType(0)),
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
                   std::vector<bool>& used)
{
  // Determine the next scale level.  This should be the first level where there
  // are any points in the far set.  So, if we know the maximum distance in the
  // distances array, this will be the largest i such that
  //   maxDistance > pow(base, i)
  // and using this for the scale factor should guarantee we are not creating an
  // implicit node.  If the maximum distance is 0, every point in the near set
  // will be created as a leaf, and a child to this node.  We also do not need
  // to change the furthestChildDistance or furthestDescendantDistance.
  const ElemType maxDistance = distances.max();
  if (maxDistance == 0)
  {
    // All points in indices are either already-used or have zero distance, so
    // we can create all the points at the lowest level (INT_MIN).
    children.push_back(new CoverTree(*dataset, base, point, INT_MIN, this, 0,
        0, distance));
    children.back()->numDescendants = 1;
    used[point] = true;

    // Every other point (which must be the same) also needs to be created as a
    // leaf.
    for (size_t i = 0; i < indices.n_elem; ++i)
    {
      if (used[indices[i]])
        continue;

      children.push_back(new CoverTree(*dataset, base, indices[i], INT_MIN,
          this, distances[i], 0, distance));
      children.back()->numDescendants = 1;
      used[indices[i]] = true;
    }

    // The number of descendants is just the number of children, because each of
    // them are leaves and contain one point.
    numDescendants = children.size();
    return;
  }

  const int nextScale = std::min(scale,
      (int) std::ceil(std::log(maxDistance) / std::log(base))) - 1;
  const ElemType bound = std::pow(base, nextScale);

  // If we got to here, then we have points in both the near set and the far
  // set.  For the self child, we need to recurse with only the near set.  For
  // other children, we must compute their distances and collect any points that
  // will be in either the near or far set.
  arma::Col<size_t> childIndices(indices.n_elem, arma::fill::none);
  arma::vec childDistances(indices.n_elem, arma::fill::none);
  size_t childSetSize = 0;
  for (size_t i = 0; i < indices.size(); ++i)
  {
    if (distances[i] >= 0 && distances[i] <= bound)
    {
      childDistances[childSetSize] = distances[i];
      childIndices[childSetSize++] = indices[i];
    }
  }

  if (childSetSize == 0)
  {
    children.push_back(new CoverTree(*dataset, base, point, INT_MIN, this, 0, 0,
        distance));
    children.back()->numDescendants = 1;
    used[point] = true;
  }
  else
  {
    // Make aliases that are shrunk to the correct size.
    arma::Col<size_t> childIndicesAlias(childIndices.memptr(), childSetSize,
        false, true);
    arma::vec childDistancesAlias(childDistances.memptr(), childSetSize, false,
        true);

    children.push_back(new CoverTree(*dataset, base, point, nextScale, this, 0,
        childIndicesAlias, childDistancesAlias, used, *distance));
  }

  numDescendants += children[0]->NumDescendants();

  // The self-child can't modify the furthestChildDistance away from 0, but it
  // can modify the furthestDescendantDistance.
  furthestDescendantDistance = children[0]->FurthestDescendantDistance();

  // Remove any implicit nodes we may have created.
  RemoveNewImplicitNodes();

  distanceComps += children[0]->DistanceComps();

  // Collect the near set points that we must make into a node.  If we are the
  // root of the tree, *all* unused points are in the near set despite what we
  // computed earlier.
  std::unordered_map<size_t, double> unusedNearSet;
  if (parent == NULL)
  {
    for (size_t i = 0; i < indices.n_elem; ++i)
      if (!used[indices[i]])
        unusedNearSet[indices[i]] = distances[i];
  }
  else
  {
    for (size_t i = 0; i < childSetSize; ++i)
      if (!used[childIndices[i]])
        unusedNearSet[childIndices[i]] = childDistances[i];
  }

  // Now for each unused point in the near set, we need to make children.
  arma::vec allChildDistances(distances.n_elem, arma::fill::none);
  while (unusedNearSet.size() > 0)
  {
    // Find the furthest distance near set point.
    size_t newPointIndex = unusedNearSet.size();
    double newPointDist = -1.0;
    for (const auto& p : unusedNearSet)
    {
      if (p.second > newPointDist)
      {
        newPointIndex = p.first;
        newPointDist = p.second;
      }
    }

    // Will this be a new furthest child?
    if (newPointDist > furthestDescendantDistance)
      furthestDescendantDistance = newPointDist;

    // Overwrite our distances array with distances from unused points to the
    // new candidate point.
    ComputeDistances(newPointIndex, indices, used, allChildDistances);

    // Create the child distances and indices.  This should reuse memory and not
    // cause an allocation.
    childSetSize = 0;
    bool childHasNearSet = false;
    for (size_t i = 0; i < indices.n_elem; ++i)
    {
      if (allChildDistances[i] >= 0 && allChildDistances[i] <= bound &&
          indices[i] != newPointIndex)
      {
        childDistances[childSetSize] = allChildDistances[i];
        childIndices[childSetSize++] = indices[i];
        if (allChildDistances[i] <= (bound / base))
          childHasNearSet = true;
      }
    }

    // If there is only one point, we can create it as a leaf, and it is the
    // last leaf we will need to create.
    if (childSetSize == 0 || !childHasNearSet)
    {
      children.push_back(new CoverTree(*dataset, base, newPointIndex, INT_MIN,
          this, newPointDist, 0, distance));
      children.back()->numDescendants = 1;
      used[newPointIndex] = true;
      unusedNearSet.erase(newPointIndex);
      ++numDescendants;
      continue;
    }

    // Create aliases that are the right size.
    arma::Col<size_t> childIndicesAlias(childIndices.memptr(), childSetSize,
        false, true);
    arma::vec childDistancesAlias(childDistances.memptr(), childSetSize, false,
        true);

    // Now recurse and build the child.
    children.push_back(new CoverTree(*dataset, base, newPointIndex, nextScale,
        this, newPointDist, childIndicesAlias, childDistancesAlias, used,
        *distance));
    numDescendants += children.back()->NumDescendants();

    // Remove any implicit nodes.
    RemoveNewImplicitNodes();
    distanceComps += children.back()->DistanceComps();

    // Now remove any points in the unused near set that we actually used.
    for (size_t i = 0; i < childSetSize; ++i)
    {
      if (used[childIndices[i]] && unusedNearSet.count(childIndices[i]) > 0)
        unusedNearSet.erase(childIndices[i]);
    }
    unusedNearSet.erase(newPointIndex);
  }

  // Calculate furthest descendant.  We can reuse the original distances here,
  // since they will be computed for any point that was not already used.
  for (size_t i = 0; i < indices.n_elem; ++i)
    if (used[indices[i]] && distances[i] > furthestDescendantDistance)
      furthestDescendantDistance = distances[i];
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
                     const std::vector<bool>& used,
                     arma::vec& distances)
{
  // For each point, rebuild the distances.  The indices do not need to be
  // modified.
  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    if (used[indices[i]])
    {
      distances[i] = -1.0;
      continue;
    }

    distances[i] = double(distance->Evaluate(dataset->col(pointIndex),
        dataset->col(indices[i])));
    ++distanceComps;
  }
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
    base(0),
    numDescendants(0),
    parent(NULL),
    parentDistance(0),
    furthestDescendantDistance(0),
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
