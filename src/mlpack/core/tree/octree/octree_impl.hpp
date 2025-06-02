/**
 * @file core/tree/octree/octree_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of generalized octree (Octree).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_OCTREE_OCTREE_IMPL_HPP
#define MLPACK_CORE_TREE_OCTREE_OCTREE_IMPL_HPP

#include "octree.hpp"
#include <mlpack/core/tree/perform_split.hpp>
#include <stack>

namespace mlpack {

//! Construct the tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(const MatType& dataset,
                                                     const size_t maxLeafSize) :
    begin(0),
    count(dataset.n_cols),
    bound(dataset.n_rows),
    dataset(new MatType(dataset)),
    parent(NULL),
    parentDistance(0.0)
{
  if (count > 0)
  {
    // Calculate empirical center of data.
    bound |= *this->dataset;
    arma::Col<ElemType> center;
    bound.Center(center);

    ElemType maxWidth = 0.0;
    for (size_t i = 0; i < bound.Dim(); ++i)
      if (bound[i].Hi() - bound[i].Lo() > maxWidth)
        maxWidth = bound[i].Hi() - bound[i].Lo();

    SplitNode(center, maxWidth, maxLeafSize);

    furthestDescendantDistance = 0.5 * bound.Diameter();
  }
  else
  {
    furthestDescendantDistance = 0.0;
  }

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Construct the tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(
    const MatType& dataset,
    std::vector<size_t>& oldFromNew,
    const size_t maxLeafSize) :
    begin(0),
    count(dataset.n_cols),
    bound(dataset.n_rows),
    dataset(new MatType(dataset)),
    parent(NULL),
    parentDistance(0.0)
{
  oldFromNew.resize(this->dataset->n_cols);
  for (size_t i = 0; i < this->dataset->n_cols; ++i)
    oldFromNew[i] = i;

  if (count > 0)
  {
    // Calculate empirical center of data.
    bound |= *this->dataset;
    arma::Col<ElemType> center;
    bound.Center(center);

    ElemType maxWidth = 0.0;
    for (size_t i = 0; i < bound.Dim(); ++i)
      if (bound[i].Hi() - bound[i].Lo() > maxWidth)
        maxWidth = bound[i].Hi() - bound[i].Lo();

    SplitNode(center, maxWidth, oldFromNew, maxLeafSize);

    furthestDescendantDistance = 0.5 * bound.Diameter();
  }
  else
  {
    furthestDescendantDistance = 0.0;
  }

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Construct the tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(
    const MatType& dataset,
    std::vector<size_t>& oldFromNew,
    std::vector<size_t>& newFromOld,
    const size_t maxLeafSize) :
    begin(0),
    count(dataset.n_cols),
    bound(dataset.n_rows),
    dataset(new MatType(dataset)),
    parent(NULL),
    parentDistance(0.0)
{
  oldFromNew.resize(this->dataset->n_cols);
  for (size_t i = 0; i < this->dataset->n_cols; ++i)
    oldFromNew[i] = i;

  if (count > 0)
  {
    // Calculate empirical center of data.
    bound |= *this->dataset;
    arma::Col<ElemType> center;
    bound.Center(center);

    ElemType maxWidth = 0.0;
    for (size_t i = 0; i < bound.Dim(); ++i)
      if (bound[i].Hi() - bound[i].Lo() > maxWidth)
        maxWidth = bound[i].Hi() - bound[i].Lo();

    SplitNode(center, maxWidth, oldFromNew, maxLeafSize);

    furthestDescendantDistance = 0.5 * bound.Diameter();
  }
  else
  {
    furthestDescendantDistance = 0.0;
  }

  // Initialize the statistic.
  stat = StatisticType(*this);

  // Map the newFromOld indices correctly.
  newFromOld.resize(this->dataset->n_cols);
  for (size_t i = 0; i < this->dataset->n_cols; ++i)
    newFromOld[oldFromNew[i]] = i;
}

//! Construct the tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(MatType&& dataset,
                                                     const size_t maxLeafSize) :
    begin(0),
    count(dataset.n_cols),
    bound(dataset.n_rows),
    dataset(new MatType(std::move(dataset))),
    parent(NULL),
    parentDistance(0.0)
{
  if (count > 0)
  {
    // Calculate empirical center of data.
    bound |= *this->dataset;
    arma::Col<ElemType> center;
    bound.Center(center);

    ElemType maxWidth = 0.0;
    for (size_t i = 0; i < bound.Dim(); ++i)
      if (bound[i].Hi() - bound[i].Lo() > maxWidth)
        maxWidth = bound[i].Hi() - bound[i].Lo();

    SplitNode(center, maxWidth, maxLeafSize);

    furthestDescendantDistance = 0.5 * bound.Diameter();
  }
  else
  {
    furthestDescendantDistance = 0.0;
  }

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Construct the tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(
    MatType&& dataset,
    std::vector<size_t>& oldFromNew,
    const size_t maxLeafSize) :
    begin(0),
    count(dataset.n_cols),
    bound(dataset.n_rows),
    dataset(new MatType(std::move(dataset))),
    parent(NULL),
    parentDistance(0.0)
{
  oldFromNew.resize(this->dataset->n_cols);
  for (size_t i = 0; i < this->dataset->n_cols; ++i)
    oldFromNew[i] = i;

  if (count > 0)
  {
    // Calculate empirical center of data.
    bound |= *this->dataset;
    arma::Col<ElemType> center;
    bound.Center(center);

    ElemType maxWidth = 0.0;
    for (size_t i = 0; i < bound.Dim(); ++i)
      if (bound[i].Hi() - bound[i].Lo() > maxWidth)
        maxWidth = bound[i].Hi() - bound[i].Lo();

    SplitNode(center, maxWidth, oldFromNew, maxLeafSize);

    furthestDescendantDistance = 0.5 * bound.Diameter();
  }
  else
  {
    furthestDescendantDistance = 0.0;
  }

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Construct the tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(
    MatType&& dataset,
    std::vector<size_t>& oldFromNew,
    std::vector<size_t>& newFromOld,
    const size_t maxLeafSize) :
    begin(0),
    count(dataset.n_cols),
    bound(dataset.n_rows),
    dataset(new MatType(std::move(dataset))),
    parent(NULL),
    parentDistance(0.0)
{
  oldFromNew.resize(this->dataset->n_cols);
  for (size_t i = 0; i < this->dataset->n_cols; ++i)
    oldFromNew[i] = i;

  if (count > 0)
  {
    // Calculate empirical center of data.
    bound |= *this->dataset;
    arma::Col<ElemType> center;
    bound.Center(center);

    ElemType maxWidth = 0.0;
    for (size_t i = 0; i < bound.Dim(); ++i)
      if (bound[i].Hi() - bound[i].Lo() > maxWidth)
        maxWidth = bound[i].Hi() - bound[i].Lo();

    SplitNode(center, maxWidth, oldFromNew, maxLeafSize);

    furthestDescendantDistance = 0.5 * bound.Diameter();
  }
  else
  {
    furthestDescendantDistance = 0.0;
  }

  // Initialize the statistic.
  stat = StatisticType(*this);

  // Map the newFromOld indices correctly.
  newFromOld.resize(this->dataset->n_cols);
  for (size_t i = 0; i < this->dataset->n_cols; ++i)
    newFromOld[oldFromNew[i]] = i;
}

//! Construct a child node.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(
    Octree* parent,
    const size_t begin,
    const size_t count,
    const arma::Col<ElemType>& center,
    const ElemType width,
    const size_t maxLeafSize) :
    begin(begin),
    count(count),
    bound(parent->dataset->n_rows),
    dataset(parent->dataset),
    parent(parent)
{
  // Calculate empirical center of data.
  bound |= dataset->cols(begin, begin + count - 1);

  // Now split the node.
  SplitNode(center, width, maxLeafSize);

  // Calculate the distance from the empirical center of this node to the
  // empirical center of the parent.
  arma::Col<ElemType> trueCenter, parentCenter;
  bound.Center(trueCenter);
  parent->Bound().Center(parentCenter);
  parentDistance = distance.Evaluate(trueCenter, parentCenter);

  furthestDescendantDistance = 0.5 * bound.Diameter();

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Construct a child node.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(
    Octree* parent,
    const size_t begin,
    const size_t count,
    std::vector<size_t>& oldFromNew,
    const arma::Col<ElemType>& center,
    const ElemType width,
    const size_t maxLeafSize) :
    begin(begin),
    count(count),
    bound(parent->dataset->n_rows),
    dataset(parent->dataset),
    parent(parent)
{
  // Calculate empirical center of data.
  bound |= dataset->cols(begin, begin + count - 1);

  // Now split the node.
  SplitNode(center, width, oldFromNew, maxLeafSize);

  // Calculate the distance from the empirical center of this node to the
  // empirical center of the parent.
  arma::Col<ElemType> trueCenter, parentCenter;
  bound.Center(trueCenter);
  parent->Bound().Center(parentCenter);
  parentDistance = distance.Evaluate(trueCenter, parentCenter);

  furthestDescendantDistance = 0.5 * bound.Diameter();

  // Initialize the statistic.
  stat = StatisticType(*this);
}

//! Copy the given tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(const Octree& other) :
    begin(other.begin),
    count(other.count),
    bound(other.bound),
    dataset((other.parent == NULL) ? new MatType(*other.dataset) : NULL),
    parent(NULL),
    stat(other.stat),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    distance(other.distance)
{
  // If we have any children, we need to create them, and then ensure that their
  // parent links are set right.
  for (size_t i = 0; i < other.NumChildren(); ++i)
  {
    children.push_back(new Octree(other.Child(i)));
    children[i]->parent = this;
    children[i]->dataset = this->dataset;
  }
}

//! Copy assignment operator: copy the given other tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>&
Octree<DistanceType, StatisticType, MatType>::
operator=(const Octree& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  delete dataset;
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  begin = other.begin;
  count = other.count;
  bound = other.bound;
  dataset = ((other.parent == NULL) ? new MatType(*other.dataset) : NULL);
  parent = NULL;
  stat = other.stat;
  parentDistance = other.ParentDistance();
  furthestDescendantDistance = other.FurthestDescendantDistance();
  distance = other.distance;

  // If we have any children, we need to create them, and then ensure that their
  // parent links are set right.
  for (size_t i = 0; i < other.NumChildren(); ++i)
  {
    children.push_back(new Octree(other.Child(i)));
    children[i]->parent = this;
    children[i]->dataset = this->dataset;
  }
  return *this;
}

//! Move the given tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree(Octree&& other) :
    children(std::move(other.children)),
    begin(other.begin),
    count(other.count),
    bound(std::move(other.bound)),
    dataset(other.dataset),
    parent(other.parent),
    stat(std::move(other.stat)),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    distance(std::move(other.distance))
{
  // Update the parent pointers of the direct children.
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->parent = this;

  other.begin = 0;
  other.count = 0;
  other.dataset = new MatType();
  other.parentDistance = 0.0;
  other.furthestDescendantDistance = 0.0;
  other.parent = NULL;
}

//! Move assignment operator: take ownership of the given tree.
template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>&
Octree<DistanceType, StatisticType, MatType>::
operator=(Octree&& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  delete dataset;
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();

  children = std::move(other.children);
  begin = other.begin;
  count = other.count;
  bound = std::move(other.bound);
  dataset = other.dataset;
  parent = other.Parent();
  stat = std::move(other.stat);
  parentDistance = other.ParentDistance();
  furthestDescendantDistance = other.furthestDescendantDistance();
  distance = std::move(other.distance);

  // Update the parent pointers of the direct children.
  for (size_t i = 0; i < children.size(); ++i)
    children[i]->parent = this;

  other.begin = 0;
  other.count = 0;
  other.dataset = new MatType();
  other.parentDistance = 0.0;
  other.furthestDescendantDistance = 0.0;
  other.parent = NULL;

  return *this;
}

template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::Octree() :
    begin(0),
    count(0),
    bound(0),
    dataset(new MatType()),
    parent(NULL),
    parentDistance(0.0),
    furthestDescendantDistance(0.0)
{
  // Nothing to do.
}

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename Archive>
Octree<DistanceType, StatisticType, MatType>::Octree(
    Archive& ar,
    const typename std::enable_if_t<cereal::is_loading<Archive>()>*) :
    Octree() // Create an empty tree.
{
  // De-serialize the tree into this object.
  ar(CEREAL_NVP(*this));
}

template<typename DistanceType, typename StatisticType, typename MatType>
Octree<DistanceType, StatisticType, MatType>::~Octree()
{
  // Delete the dataset if we aren't the parent.
  if (!parent)
    delete dataset;

  // Now delete each of the children.
  for (size_t i = 0; i < children.size(); ++i)
    delete children[i];
  children.clear();
}

template<typename DistanceType, typename StatisticType, typename MatType>
size_t Octree<DistanceType, StatisticType, MatType>::NumChildren() const
{
  return children.size();
}

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename VecType>
size_t Octree<DistanceType, StatisticType, MatType>::GetNearestChild(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  // It's possible that this could be improved by caching which children we have
  // and which we don't, but for now this is just a brute force search.
  ElemType bestDistance = DBL_MAX;
  size_t bestIndex = NumChildren();
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    const ElemType dist = children[i]->MinDistance(point);
    if (dist < bestDistance)
    {
      bestDistance = dist;
      bestIndex = i;
    }
  }

  return bestIndex;
}

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename VecType>
size_t Octree<DistanceType, StatisticType, MatType>::GetFurthestChild(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  // It's possible that this could be improved by caching which children we have
  // and which we don't, but for now this is just a brute force search.
  ElemType bestDistance = -1.0; // Initialize to invalid distance.
  size_t bestIndex = NumChildren();
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    const ElemType dist = children[i]->MaxDistance(point);
    if (dist > bestDistance)
    {
      bestDistance = dist;
      bestIndex = i;
    }
  }

  return bestIndex;
}

template<typename DistanceType, typename StatisticType, typename MatType>
size_t Octree<DistanceType, StatisticType, MatType>::GetNearestChild(
    const Octree& queryNode) const
{
  // It's possible that this could be improved by caching which children we have
  // and which we don't, but for now this is just a brute force search.
  ElemType bestDistance = DBL_MAX;
  size_t bestIndex = NumChildren();
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    const ElemType dist = children[i]->MinDistance(queryNode);
    if (dist < bestDistance)
    {
      bestDistance = dist;
      bestIndex = i;
    }
  }

  return bestIndex;
}

template<typename DistanceType, typename StatisticType, typename MatType>
size_t Octree<DistanceType, StatisticType, MatType>::GetFurthestChild(
    const Octree& queryNode) const
{
  // It's possible that this could be improved by caching which children we have
  // and which we don't, but for now this is just a brute force search.
  ElemType bestDistance = -1.0; // Initialize to invalid distance.
  size_t bestIndex = NumChildren();
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    const ElemType dist = children[i]->MaxDistance(queryNode);
    if (dist > bestDistance)
    {
      bestDistance = dist;
      bestIndex = i;
    }
  }

  return bestIndex;
}

template<typename DistanceType, typename StatisticType, typename MatType>
typename Octree<DistanceType, StatisticType, MatType>::ElemType
Octree<DistanceType, StatisticType, MatType>::FurthestPointDistance()
    const
{
  // If we are not a leaf, then this distance is 0.  Otherwise, return the
  // furthest descendant distance.
  return (children.size() > 0) ? 0.0 : furthestDescendantDistance;
}

template<typename DistanceType, typename StatisticType, typename MatType>
typename Octree<DistanceType, StatisticType, MatType>::ElemType
Octree<DistanceType, StatisticType, MatType>::FurthestDescendantDistance() const
{
  return furthestDescendantDistance;
}

template<typename DistanceType, typename StatisticType, typename MatType>
typename Octree<DistanceType, StatisticType, MatType>::ElemType
Octree<DistanceType, StatisticType, MatType>::MinimumBoundDistance() const
{
  return bound.MinWidth() / 2.0;
}

template<typename DistanceType, typename StatisticType, typename MatType>
size_t Octree<DistanceType, StatisticType, MatType>::NumPoints() const
{
  // We have no points unless we are a leaf;
  return (children.size() > 0) ? 0 : count;
}

template<typename DistanceType, typename StatisticType, typename MatType>
size_t Octree<DistanceType, StatisticType, MatType>::NumDescendants() const
{
  return count;
}

template<typename DistanceType, typename StatisticType, typename MatType>
size_t Octree<DistanceType, StatisticType, MatType>::Descendant(
    const size_t index) const
{
  return begin + index;
}

template<typename DistanceType, typename StatisticType, typename MatType>
size_t Octree<DistanceType, StatisticType, MatType>::Point(const size_t index)
    const
{
  return begin + index;
}

template<typename DistanceType, typename StatisticType, typename MatType>
typename Octree<DistanceType, StatisticType, MatType>::ElemType
Octree<DistanceType, StatisticType, MatType>::MinDistance(const Octree& other)
    const
{
  return bound.MinDistance(other.Bound());
}

template<typename DistanceType, typename StatisticType, typename MatType>
typename Octree<DistanceType, StatisticType, MatType>::ElemType
Octree<DistanceType, StatisticType, MatType>::MaxDistance(const Octree& other)
    const
{
  return bound.MaxDistance(other.Bound());
}

template<typename DistanceType, typename StatisticType, typename MatType>
RangeType<typename Octree<DistanceType, StatisticType, MatType>::ElemType>
Octree<DistanceType, StatisticType, MatType>::RangeDistance(const Octree& other)
    const
{
  return bound.RangeDistance(other.Bound());
}

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename VecType>
typename Octree<DistanceType, StatisticType, MatType>::ElemType
Octree<DistanceType, StatisticType, MatType>::MinDistance(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  return bound.MinDistance(point);
}

template<typename DistanceType, typename StatisticType, typename MatType>
template<typename VecType>
typename Octree<DistanceType, StatisticType, MatType>::ElemType
Octree<DistanceType, StatisticType, MatType>::MaxDistance(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  return bound.MaxDistance(point);
}


template<typename DistanceType, typename StatisticType, typename MatType>
template<typename VecType>
RangeType<typename Octree<DistanceType, StatisticType, MatType>::ElemType>
Octree<DistanceType, StatisticType, MatType>::RangeDistance(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*) const
{
  return bound.RangeDistance(point);
}

//! Serialize the tree.
template<typename DistanceType, typename StatisticType, typename MatType>
template<typename Archive>
void Octree<DistanceType, StatisticType, MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  // If we're loading and we have children, they need to be deleted.
  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < children.size(); ++i)
      delete children[i];
    children.clear();

    if (!parent)
      delete dataset;

    parent = NULL;
  }

  bool hasParent = (parent != NULL);

  ar(CEREAL_NVP(begin));
  ar(CEREAL_NVP(count));
  ar(CEREAL_NVP(bound));
  ar(CEREAL_NVP(stat));
  ar(CEREAL_NVP(parentDistance));
  ar(CEREAL_NVP(furthestDescendantDistance));
  ar(CEREAL_NVP(distance));
  ar(CEREAL_NVP(hasParent));
  if (!hasParent)
  {
    MatType*& datasetTemp = const_cast<MatType*&>(dataset);
    ar(CEREAL_POINTER(datasetTemp));
  }

  ar(CEREAL_VECTOR_POINTER(children));

  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < children.size(); ++i)
      children[i]->parent = this;
  }

  // We have to correct the dataset pointers in all of the children.
  if (!hasParent)
  {
    std::stack<Octree*> stack;
    for (size_t i = 0; i < children.size(); ++i)
    {
      stack.push(children[i]);
    }
    while (!stack.empty())
    {
      Octree* node = stack.top();
      stack.pop();
      node->dataset = dataset;
      for (size_t i = 0; i < node->children.size(); ++i)
      {
        stack.push(node->children[i]);
      }
    }
  }
}

//! Split the node.
template<typename DistanceType, typename StatisticType, typename MatType>
void Octree<DistanceType, StatisticType, MatType>::SplitNode(
    const arma::Col<ElemType>& center,
    const ElemType width,
    const size_t maxLeafSize)
{
  // No need to split if we have fewer than the maximum number of points in this
  // node.
  if (count <= maxLeafSize)
    return;

  // This will hold the index of the first point in each child.
  arma::Col<size_t> childBegins(((size_t) 1 << dataset->n_rows) + 1);
  childBegins[0] = begin;
  childBegins[childBegins.n_elem - 1] = begin + count;

  // We will make log2(dim) passes, splitting along the last down to the first
  // dimension.  The tuple holds { dim, begin, count, leftChildIndex }.
  std::stack<std::tuple<size_t, size_t, size_t, size_t>> stack;
  stack.push(std::tuple<size_t, size_t, size_t, size_t>(dataset->n_rows - 1,
      begin, count, 0));

  while (!stack.empty())
  {
    std::tuple<size_t, size_t, size_t, size_t> t = stack.top();
    stack.pop();

    const size_t d = std::get<0>(t);
    const size_t childBegin = std::get<1>(t);
    const size_t childCount = std::get<2>(t);
    const size_t leftChildIndex = std::get<3>(t);

    // Perform a "half-split": after this split, all points belonging to
    // children of index 2^(d - 1) - 1 and less will be on the left side, and
    // all points belonging to children of index 2^(d - 1) and above will be on
    // the right side.
    typename SplitType::SplitInfo s(d, center);
    const size_t firstRight = PerformSplit<MatType, SplitType>(*dataset,
        childBegin, childCount, s);

    // We can set the first index of the right child.  The first index of the
    // left child is already set.
    const size_t rightChildIndex = leftChildIndex + ((size_t) 1 << d);
    childBegins[rightChildIndex] = firstRight;

    // Now we have to recurse, if this was not the last dimension.
    if (d != 0)
    {
      if (firstRight > childBegin)
      {
        stack.push(std::tuple<size_t, size_t, size_t, size_t>(d - 1, childBegin,
            firstRight - childBegin, leftChildIndex));
      }
      else
      {
        // Set beginning indices correctly for all children below this level.
        for (size_t c = leftChildIndex + 1; c < rightChildIndex; ++c)
          childBegins[c] = childBegins[leftChildIndex];
      }

      if (firstRight < childBegin + childCount)
      {
        stack.push(std::tuple<size_t, size_t, size_t, size_t>(d - 1, firstRight,
            childCount - (firstRight - childBegin), rightChildIndex));
      }
      else
      {
        // Set beginning indices correctly for all children below this level.
        for (size_t c = rightChildIndex + 1;
             c < rightChildIndex + (rightChildIndex - leftChildIndex); ++c)
          childBegins[c] = childBegins[rightChildIndex];
      }
    }
  }

  // Now that the dataset is reordered, we can create the children.
  arma::Col<ElemType> childCenter(center.n_elem);
  const ElemType childWidth = width / 2.0;
  for (size_t i = 0; i < childBegins.n_elem - 1; ++i)
  {
    // If the child has no points, don't create it.
    if (childBegins[i + 1] - childBegins[i] == 0)
      continue;

    // Create the correct center.
    for (size_t d = 0; d < center.n_elem; ++d)
    {
      // Is the dimension "right" (1) or "left" (0)?
      if (((i >> d) & 1) == 0)
        childCenter[d] = center[d] - childWidth;
      else
        childCenter[d] = center[d] + childWidth;
    }

    children.push_back(new Octree(this, childBegins[i],
        childBegins[i + 1] - childBegins[i], childCenter, childWidth,
        maxLeafSize));
  }
}

//! Split the node, and store mappings.
template<typename DistanceType, typename StatisticType, typename MatType>
void Octree<DistanceType, StatisticType, MatType>::SplitNode(
    const arma::Col<ElemType>& center,
    const ElemType width,
    std::vector<size_t>& oldFromNew,
    const size_t maxLeafSize)
{
  // No need to split if we have fewer than the maximum number of points in this
  // node.
  if (count <= maxLeafSize)
    return;

  // This will hold the index of the first point in each child.
  arma::Col<size_t> childBegins(((size_t) 1 << dataset->n_rows) + 1);
  childBegins[0] = begin;
  childBegins[childBegins.n_elem - 1] = begin + count;

  // We will make log2(dim) passes, splitting along the last down to the first
  // dimension.  The tuple holds { dim, begin, count, leftChildIndex }.
  std::stack<std::tuple<size_t, size_t, size_t, size_t>> stack;
  stack.push(std::tuple<size_t, size_t, size_t, size_t>(dataset->n_rows - 1,
      begin, count, 0));

  while (!stack.empty())
  {
    std::tuple<size_t, size_t, size_t, size_t> t = stack.top();
    stack.pop();

    const size_t d = std::get<0>(t);
    const size_t childBegin = std::get<1>(t);
    const size_t childCount = std::get<2>(t);
    const size_t leftChildIndex = std::get<3>(t);

    // Perform a "half-split": after this split, all points belonging to
    // children of index 2^(d - 1) - 1 and less will be on the left side, and
    // all points belonging to children of index 2^(d - 1) and above will be on
    // the right side.
    typename SplitType::SplitInfo s(d, center);
    const size_t firstRight = PerformSplit<MatType, SplitType>(*dataset,
        childBegin, childCount, s, oldFromNew);

    // We can set the first index of the right child.  The first index of the
    // left child is already set.
    const size_t rightChildIndex = leftChildIndex + ((size_t) 1 << d);
    childBegins[rightChildIndex] = firstRight;

    // Now we have to recurse, if this was not the last dimension.
    if (d != 0)
    {
      if (firstRight > childBegin)
      {
        stack.push(std::tuple<size_t, size_t, size_t, size_t>(d - 1, childBegin,
            firstRight - childBegin, leftChildIndex));
      }
      else
      {
        // Set beginning indices correctly for all children below this level.
        for (size_t c = leftChildIndex + 1; c < rightChildIndex; ++c)
          childBegins[c] = childBegins[leftChildIndex];
      }

      if (firstRight < childBegin + childCount)
      {
        stack.push(std::tuple<size_t, size_t, size_t, size_t>(d - 1, firstRight,
            childCount - (firstRight - childBegin), rightChildIndex));
      }
      else
      {
        // Set beginning indices correctly for all children below this level.
        for (size_t c = rightChildIndex + 1;
             c < rightChildIndex + (rightChildIndex - leftChildIndex); ++c)
          childBegins[c] = childBegins[rightChildIndex];
      }
    }
  }

  // Now that the dataset is reordered, we can create the children.
  arma::Col<ElemType> childCenter(center.n_elem);
  const ElemType childWidth = width / 2.0;
  for (size_t i = 0; i < childBegins.n_elem - 1; ++i)
  {
    // If the child has no points, don't create it.
    if (childBegins[i + 1] - childBegins[i] == 0)
      continue;

    // Create the correct center.
    for (size_t d = 0; d < center.n_elem; ++d)
    {
      // Is the dimension "right" (1) or "left" (0)?
      if (((i >> d) & 1) == 0)
        childCenter[d] = center[d] - childWidth;
      else
        childCenter[d] = center[d] + childWidth;
    }

    children.push_back(new Octree(this, childBegins[i],
        childBegins[i + 1] - childBegins[i], oldFromNew, childCenter,
        childWidth, maxLeafSize));
  }
}

} // namespace mlpack

#endif
