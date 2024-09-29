/**
 * @file core/tree/rectangle_tree/rectangle_tree_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of generalized rectangle tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "rectangle_tree.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

// Build the statistics, bottom-up.
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
BuildStatistics(RectangleTree* node)
{
  // Recurse first.
  for (size_t i = 0; i < node->NumChildren(); ++i)
    BuildStatistics(&node->Child(i));

  // Now build the statistic.
  node->Stat() = StatisticType(*node);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
RectangleTree(const MatType& data,
              const size_t maxLeafSize,
              const size_t minLeafSize,
              const size_t maxNumChildren,
              const size_t minNumChildren,
              const size_t firstDataIndex) :
    maxNumChildren(maxNumChildren),
    minNumChildren(minNumChildren),
    numChildren(0),
    children(maxNumChildren + 1), // Add one to make splitting the node simpler.
    parent(NULL),
    begin(0),
    count(0),
    numDescendants(0),
    maxLeafSize(maxLeafSize),
    minLeafSize(minLeafSize),
    bound(data.n_rows),
    parentDistance(0),
    dataset(new MatType(data)),
    ownsDataset(true),
    points(maxLeafSize + 1), // Add one to make splitting the node simpler.
    auxiliaryInfo(this)
{
  // For now, just insert the points in order.
  RectangleTree* root = this;

  for (size_t i = firstDataIndex; i < data.n_cols; ++i)
    root->InsertPoint(i);

  // Initialize statistic recursively after tree construction is complete.
  BuildStatistics(this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
RectangleTree(MatType&& data,
              const size_t maxLeafSize,
              const size_t minLeafSize,
              const size_t maxNumChildren,
              const size_t minNumChildren,
              const size_t firstDataIndex) :
    maxNumChildren(maxNumChildren),
    minNumChildren(minNumChildren),
    numChildren(0),
    children(maxNumChildren + 1), // Add one to make splitting the node simpler.
    parent(NULL),
    begin(0),
    count(0),
    numDescendants(0),
    maxLeafSize(maxLeafSize),
    minLeafSize(minLeafSize),
    bound(data.n_rows),
    parentDistance(0),
    dataset(new MatType(std::move(data))),
    ownsDataset(true),
    points(maxLeafSize + 1), // Add one to make splitting the node simpler.
    auxiliaryInfo(this)
{
  // For now, just insert the points in order.
  RectangleTree* root = this;

  for (size_t i = firstDataIndex; i < dataset->n_cols; ++i)
    root->InsertPoint(i);

  // Initialize statistic recursively after tree construction is complete.
  BuildStatistics(this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
RectangleTree(
    RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                  AuxiliaryInformationType>*
        parentNode, const size_t numMaxChildren) :
    maxNumChildren(numMaxChildren > 0 ? numMaxChildren :
                                        parentNode->MaxNumChildren()),
    minNumChildren(parentNode->MinNumChildren()),
    numChildren(0),
    children(maxNumChildren + 1),
    parent(parentNode),
    begin(0),
    count(0),
    numDescendants(0),
    maxLeafSize(parentNode->MaxLeafSize()),
    minLeafSize(parentNode->MinLeafSize()),
    bound(parentNode->Bound().Dim()),
    parentDistance(0),
    dataset(&parentNode->Dataset()),
    ownsDataset(false),
    points(maxLeafSize + 1), // Add one to make splitting the node simpler.
    auxiliaryInfo(this)
{
  // Initialize statistic.
  BuildStatistics(this);
}

/**
 * Create a rectangle tree by copying the other tree.  Be careful!  This can
 * take a long time and use a lot of memory.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
RectangleTree(
    const RectangleTree& other,
    const bool deepCopy,
    RectangleTree* newParent) :
    maxNumChildren(other.MaxNumChildren()),
    minNumChildren(other.MinNumChildren()),
    numChildren(other.NumChildren()),
    children(maxNumChildren + 1, NULL),
    parent(deepCopy ? newParent : other.Parent()),
    begin(other.Begin()),
    count(other.Count()),
    numDescendants(other.numDescendants),
    maxLeafSize(other.MaxLeafSize()),
    minLeafSize(other.MinLeafSize()),
    bound(other.bound),
    stat(other.stat),
    parentDistance(other.ParentDistance()),
    dataset(deepCopy ?
        (parent ? parent->dataset : new MatType(*other.dataset)) :
        &other.Dataset()),
    ownsDataset(deepCopy && (!parent)),
    points(other.points),
    auxiliaryInfo(other.auxiliaryInfo, this, deepCopy)
{
  if (deepCopy)
  {
    if (numChildren > 0)
    {
      for (size_t i = 0; i < numChildren; ++i)
        children[i] = new RectangleTree(other.Child(i), true, this);
    }
  }
  else
    children = other.children;
}

/**
 * Move constructor.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
RectangleTree(RectangleTree&& other) :
    maxNumChildren(other.MaxNumChildren()),
    minNumChildren(other.MinNumChildren()),
    numChildren(other.NumChildren()),
    children(std::move(other.children)),
    parent(other.Parent()),
    begin(other.Begin()),
    count(other.Count()),
    numDescendants(other.numDescendants),
    maxLeafSize(other.MaxLeafSize()),
    minLeafSize(other.MinLeafSize()),
    bound(std::move(other.bound)),
    stat(std::move(other.stat)),
    parentDistance(other.ParentDistance()),
    dataset(other.dataset),
    ownsDataset(other.ownsDataset),
    points(std::move(other.points)),
    auxiliaryInfo(std::move(other.auxiliaryInfo))
{
  if (parent)
  {
    size_t iChild = 0;
    while (parent->children[iChild] != (&other))
      iChild++;
    assert(iChild < numChildren);
    parent->children[iChild] = this;
  }
  if (!IsLeaf())
  {
    for (size_t i = 0; i < numChildren; ++i)
      children[i]->parent = this;
  }
  // Now we are a clone of the other tree.  But we must also clear the other
  // tree's contents, so it doesn't delete anything when it is destructed.
  other.maxNumChildren = 0;
  other.minNumChildren = 0;
  other.numChildren = 0;
  other.parent = NULL;
  other.begin = 0;
  other.count = 0;
  other.numDescendants = 0;
  other.maxLeafSize = 0;
  other.minLeafSize = 0;
  other.parentDistance = 0;
  other.dataset = NULL;
  other.ownsDataset = false;
}

/**
 * Copy assignment operator: copy the given other tree.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>&
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
operator=(const RectangleTree& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  for (size_t i = 0; i < numChildren; ++i)
    delete children[i];

  if (ownsDataset)
    delete dataset;

  maxNumChildren = other.MaxNumChildren();
  minNumChildren = other.MinNumChildren();
  numChildren = other.NumChildren();
  children.resize(maxNumChildren + 1, NULL);
  parent = NULL;
  begin = other.Begin();
  count = other.Count();
  numDescendants = other.numDescendants;
  maxLeafSize = other.MaxLeafSize();
  minLeafSize = other.MinLeafSize();
  bound = other.bound;
  stat = other.stat;
  parentDistance = other.ParentDistance();
  dataset = new MatType(*other.dataset);
  ownsDataset = true;
  points = other.points;
  auxiliaryInfo = AuxiliaryInfoType(other.auxiliaryInfo, this, true);

  if (numChildren > 0)
  {
    for (size_t i = 0; i < numChildren; ++i)
      children[i] = new RectangleTree(other.Child(i), true, this);
  }

  return *this;
}

/**
 * Move assignment operator: take ownership of the given tree.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>&
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
operator=(RectangleTree&& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  for (size_t i = 0; i < numChildren; ++i)
    delete children[i];

  if (ownsDataset)
    delete dataset;

  maxNumChildren = other.MaxNumChildren();
  minNumChildren = other.MinNumChildren();
  numChildren = other.NumChildren();
  children = std::move(other.children);
  parent = other.Parent();
  begin = other.Begin();
  count = other.Count();
  numDescendants = other.numDescendants;
  maxLeafSize = other.MaxLeafSize();
  minLeafSize = other.MinLeafSize();
  bound = std::move(other.bound);
  stat = std::move(other.stat);
  parentDistance = other.ParentDistance();
  dataset = other.dataset;
  ownsDataset = other.ownsDataset;
  points = std::move(other.points);
  auxiliaryInfo = std::move(other.auxiliaryInfo);

  // Now we are a clone of the other tree.  But we must also clear the other
  // tree's contents, so it doesn't delete anything when it is destructed.
  other.maxNumChildren = 0;
  other.minNumChildren = 0;
  other.numChildren = 0;
  other.parent = NULL;
  other.begin = 0;
  other.count = 0;
  other.numDescendants = 0;
  other.maxLeafSize = 0;
  other.minLeafSize = 0;
  other.parentDistance = 0;
  other.dataset = NULL;
  other.ownsDataset = false;

  return *this;
}

/**
 * Construct the tree from a cereal archive.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename Archive>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
RectangleTree(
    Archive& ar,
    const typename std::enable_if_t<cereal::is_loading<Archive>()>*) :
    RectangleTree() // Use default constructor.
{
  // Now serialize.
  ar(CEREAL_NVP(*this));
}

/**
 * Deletes this node, deallocating the memory for the children and calling
 * their destructors in turn.  This will invalidate any pointers or references
 * to any nodes which are children of this one.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
~RectangleTree()
{
  for (size_t i = 0; i < numChildren; ++i)
    delete children[i];

  if (ownsDataset)
    delete dataset;
}

/**
 * Deletes this node but leaves the children untouched.  Needed for when we
 * split nodes and remove nodes (inserting and deleting points).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    SoftDelete()
{
  parent = NULL;

  for (size_t i = 0; i < children.size(); ++i)
    children[i] = NULL;

  numChildren = 0;
  delete this;
}

/**
 * Nullify the auxiliary information.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    NullifyData()
{
  auxiliaryInfo.NullifyData();
}

/**
 * Recurse through the tree and insert the point at the leaf node chosen
 * by the heuristic.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    InsertPoint(const size_t point)
{
  // Expand the bound regardless of whether it is a leaf node.
  bound |= dataset->col(point);

  numDescendants++;

  std::vector<bool> lvls(TreeDepth(), true);

  // If this is a leaf node, we stop here and add the point.
  if (numChildren == 0)
  {
    if (!auxiliaryInfo.HandlePointInsertion(this, point))
      points[count++] = point;

    SplitNode(lvls);
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  auxiliaryInfo.HandlePointInsertion(this, point);
  const size_t descentNode = DescentType::ChooseDescentNode(this, point);
  children[descentNode]->InsertPoint(point, lvls);
}

/**
 * Inserts a point into the tree, tracking which levels have been inserted into.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    InsertPoint(const size_t point, std::vector<bool>& relevels)
{
  // Expand the bound regardless of whether it is a leaf node.
  bound |= dataset->col(point);

  numDescendants++;

  // If this is a leaf node, we stop here and add the point.
  if (numChildren == 0)
  {
    if (!auxiliaryInfo.HandlePointInsertion(this, point))
      points[count++] = point;

    SplitNode(relevels);
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  auxiliaryInfo.HandlePointInsertion(this, point);
  const size_t descentNode = DescentType::ChooseDescentNode(this, point);
  children[descentNode]->InsertPoint(point, relevels);
}

/**
 * Inserts a node into the tree, tracking which levels have been inserted into.
 *
 * @param node The node to be inserted.
 * @param level The level on which this node should be inserted.
 * @param relevels The levels that have been reinserted to on this top level
 *      insertion.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    InsertNode(RectangleTree* node,
               const size_t level,
               std::vector<bool>& relevels)
{
  // Expand the bound regardless of the level.
  bound |= node->Bound();
  numDescendants += node->numDescendants;
  if (level == TreeDepth())
  {
    if (!auxiliaryInfo.HandleNodeInsertion(this, node, true))
    {
      children[numChildren++] = node;
      node->Parent() = this;
    }
    SplitNode(relevels);
  }
  else
  {
    auxiliaryInfo.HandleNodeInsertion(this, node, false);
    const size_t descentNode = DescentType::ChooseDescentNode(this, node);
    children[descentNode]->InsertNode(node, level, relevels);
  }
}

/**
 * Recurse through the tree to remove the point.  Once we find the point, we
 * shrink the rectangles if necessary.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
bool RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    DeletePoint(const size_t point)
{
  // It is possible that this will cause a reinsertion, so we need to handle the
  // levels properly.
  RectangleTree* root = this;
  while (root->Parent() != NULL)
    root = root->Parent();

  std::vector<bool> lvls(root->TreeDepth(), true);

  if (numChildren == 0)
  {
    for (size_t i = 0; i < count; ++i)
    {
      if (points[i] == point)
      {
        if (!auxiliaryInfo.HandlePointDeletion(this, i))
          points[i] = points[--count];

        RectangleTree* tree = this;
        while (tree != NULL)
        {
          tree->numDescendants--;
          tree = tree->Parent();
        }
        // This function wil ensure that minFill is satisfied.
        CondenseTree(dataset->col(point), lvls, true);
        return true;
      }
    }
  }

  for (size_t i = 0; i < numChildren; ++i)
    if (children[i]->Bound().Contains(dataset->col(point)))
      if (children[i]->DeletePoint(point, lvls))
        return true;

  return false;
}

/**
 * Recurse through the tree to remove the point.  Once we find the point, we
 * shrink the rectangles if necessary.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
bool RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    DeletePoint(const size_t point, std::vector<bool>& relevels)
{
  if (numChildren == 0)
  {
    for (size_t i = 0; i < count; ++i)
    {
      if (points[i] == point)
      {
        if (!auxiliaryInfo.HandlePointDeletion(this, i))
          points[i] = points[--count];

        RectangleTree* tree = this;
        while (tree != NULL)
        {
          tree->numDescendants--;
          tree = tree->Parent();
        }
        // This function will ensure that minFill is satisfied.
        CondenseTree(dataset->col(point), relevels, true);
        return true;
      }
    }
  }

  for (size_t i = 0; i < numChildren; ++i)
    if (children[i]->Bound().Contains(dataset->col(point)))
      if (children[i]->DeletePoint(point, relevels))
        return true;

  return false;
}


/**
 * Recurse through the tree to remove the node.  Once we find the node, we
 * shrink the rectangles if necessary.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
bool RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    RemoveNode(const RectangleTree* node, std::vector<bool>& relevels)
{
  for (size_t i = 0; i < numChildren; ++i)
  {
    if (children[i] == node)
    {
      if (!auxiliaryInfo.HandleNodeRemoval(this, i))
      {
        children[i] = children[--numChildren]; // Decrement numChildren.
      }
      RectangleTree* tree = this;
      while (tree != NULL)
      {
        tree->numDescendants -= node->numDescendants;
        tree = tree->Parent();
      }
      CondenseTree(arma::vec(), relevels, false);
      return true;
    }

    bool contains = true;
    for (size_t j = 0; j < node->Bound().Dim(); ++j)
      contains &= Child(i).Bound()[j].Contains(node->Bound()[j]);

    if (contains)
      if (children[i]->RemoveNode(node, relevels))
        return true;
  }

  return false;
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType,
                     DescentType, AuxiliaryInformationType>::TreeSize() const
{
  int n = 0;
  for (int i = 0; i < numChildren; ++i)
    n += children[i]->TreeSize();

  return n + 1; // Add one for this node.
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType,
                     DescentType, AuxiliaryInformationType>::TreeDepth() const
{
  int n = 1;
  RectangleTree* currentNode = const_cast<RectangleTree*> (this);

  while (!currentNode->IsLeaf())
  {
    currentNode = currentNode->children[0];
    n++;
  }

  return n;
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
inline bool RectangleTree<DistanceType, StatisticType, MatType, SplitType,
                          DescentType, AuxiliaryInformationType>::IsLeaf() const
{
  return (numChildren == 0);
}

/**
 * Return the index of the nearest child node to the given query point.  If
 * this is a leaf node, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename VecType>
size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
    AuxiliaryInformationType>::GetNearestChild(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    ElemType distance = Child(i).MinDistance(point);
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
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename VecType>
size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
    AuxiliaryInformationType>::GetFurthestChild(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = 0;
  size_t bestIndex = 0;
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    ElemType distance = Child(i).MaxDistance(point);
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
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
    AuxiliaryInformationType>::GetNearestChild(const RectangleTree& queryNode)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    ElemType distance = Child(i).MinDistance(queryNode);
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
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
    AuxiliaryInformationType>::GetFurthestChild(const RectangleTree& queryNode)
{
  if (IsLeaf())
    return 0;

  ElemType bestDistance = 0;
  size_t bestIndex = 0;
  for (size_t i = 0; i < NumChildren(); ++i)
  {
    ElemType distance = Child(i).MaxDistance(queryNode);
    if (distance >= bestDistance)
    {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Return a bound on the furthest point in the node form the centroid.
 * This returns 0 unless the node is a leaf.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
inline
typename RectangleTree<DistanceType, StatisticType, MatType, SplitType,
    DescentType, AuxiliaryInformationType>::ElemType
RectangleTree<DistanceType, StatisticType, MatType, SplitType,
    DescentType, AuxiliaryInformationType>::FurthestPointDistance() const
{
  if (!IsLeaf())
    return 0.0;

  // Otherwise return the distance from the centroid to a corner of the bound.
  return 0.5 * bound.Diameter();
}

/**
 * Return the furthest possible descendant distance.  This returns the maximum
 * distance from the centroid to the edge of the bound and not the empirical
 * quantity which is the actual furthest descendant distance.  So the actual
 * furthest descendant distance may be less than what this method returns (but
 * it will never be greater than this).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
inline
typename RectangleTree<DistanceType, StatisticType, MatType, SplitType,
    DescentType, AuxiliaryInformationType>::ElemType
RectangleTree<DistanceType, StatisticType, MatType, SplitType,
    DescentType, AuxiliaryInformationType>::FurthestDescendantDistance() const
{
  // Return the distance from the centroid to a corner of the bound.
  return 0.5 * bound.Diameter();
}

/**
 * Return the number of points contained in this node.  Zero if it is a non-leaf
 * node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
inline size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType,
                       DescentType, AuxiliaryInformationType>::NumPoints() const
{
  if (numChildren != 0) // This is not a leaf node.
    return 0;

  return count;
}

/**
 * Return the number of descendants under or in this node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
inline size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType,
                  DescentType, AuxiliaryInformationType>::NumDescendants() const
{
  return numDescendants;
}

/**
 * Return the index of a particular descendant contained in this node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
inline size_t RectangleTree<DistanceType, StatisticType, MatType, SplitType,
    DescentType, AuxiliaryInformationType>::Descendant(const size_t index) const
{
  // I think this may be inefficient...
  if (numChildren == 0)
  {
    return (points[index]);
  }
  else
  {
    size_t n = 0;
    for (size_t i = 0; i < numChildren; ++i)
    {
      const size_t nd = children[i]->NumDescendants();
      if (index - n < nd)
        return children[i]->Descendant(index - n);
      n += nd;
    }

    // I don't think this is valid.
    return children[numChildren - 1]->Descendant(index - n);
  }
}

/**
 * Split the tree.  This calls the SplitType code to split a node.  This method
 * should only be called on a leaf node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    SplitNode(std::vector<bool>& relevels)
{
  if (numChildren == 0)
  {
    // We let the SplitType check if the node if overflowed
    // since an intermediate node of the R+ tree may be overflowed if the leaf
    // node contains only one point.

    // The SplitType takes care of this and of moving up the tree if necessary.
    SplitType::SplitLeafNode(this, relevels);
  }
  else
  {
    // Check to see if we are full.
    if (numChildren <= maxNumChildren)
      return; // We don't need to split.

    // If we are full, then we need to split (or at least try).  The SplitType
    // takes care of this and of moving up the tree if necessary.
    SplitType::SplitNonLeafNode(this, relevels);
  }
}

//! Default constructor for cereal.
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
              AuxiliaryInformationType>::
RectangleTree() :
    maxNumChildren(0), // Try to give sensible defaults, but it shouldn't matter
    minNumChildren(0), // because this tree isn't valid anyway and is only used
    numChildren(0),    // by cereal.
    parent(NULL),
    begin(0),
    count(0),
    numDescendants(0),
    maxLeafSize(0),
    minLeafSize(0),
    parentDistance(0.0),
    dataset(NULL),
    ownsDataset(false)
{
  // Nothing to do.
}

/**
 * Condense the tree.  This shrinks the bounds and moves up the tree if
 * applicable.  If a node goes below minimum fill, this code will deal with it.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    CondenseTree(const arma::vec& point,
                 std::vector<bool>& relevels,
                 const bool usePoint)
{
  // First delete the node if we need to.  There's no point in shrinking the
  // bound first.
  if (IsLeaf() && count < minLeafSize && parent != NULL)
  {
    // We can't delete the root node.
    for (size_t i = 0; i < parent->NumChildren(); ++i)
    {
      if (parent->children[i] == this)
      {
        // Decrement numChildren.
        if (!auxiliaryInfo.HandleNodeRemoval(parent, i))
        {
          parent->children[i] = parent->children[--parent->NumChildren()];
        }

        // We find the root and shrink bounds at the same time.
        bool stillShrinking = true;
        RectangleTree* root = parent;
        while (root->Parent() != NULL)
        {
          if (stillShrinking)
            stillShrinking = root->ShrinkBoundForBound(bound);
          root = root->Parent();
        }
        if (stillShrinking)
          root->ShrinkBoundForBound(bound);

        root = parent;
        while (root != NULL)
        {
          root->numDescendants -= numDescendants;
          root = root->Parent();
        }

        stillShrinking = true;
        root = parent;
        while (root->Parent() != NULL)
        {
          if (stillShrinking)
            stillShrinking = root->AuxiliaryInfo().UpdateAuxiliaryInfo(root);
          root = root->Parent();
        }
        if (stillShrinking)
          root->AuxiliaryInfo().UpdateAuxiliaryInfo(root);

       // Reinsert the points at the root node.
        for (size_t j = 0; j < count; ++j)
          root->InsertPoint(points[j], relevels);

        // This will check the minFill of the parent.
        parent->CondenseTree(point, relevels, usePoint);
        // Now it should be safe to delete this node.
        SoftDelete();

        return;
      }
    }
    // Control should never reach here.
    assert(false);
  }
  else if (!IsLeaf() && numChildren < minNumChildren)
  {
    if (parent != NULL)
    {
      // The normal case.  We need to be careful with the root.
      for (size_t j = 0; j < parent->NumChildren(); ++j)
      {
        if (parent->children[j] == this)
        {
          // Decrement numChildren.
          if (!auxiliaryInfo.HandleNodeRemoval(parent, j))
          {
            parent->children[j] = parent->children[--parent->NumChildren()];
          }
          size_t level = TreeDepth();

          // We find the root and shrink bounds at the same time.
          bool stillShrinking = true;
          RectangleTree* root = parent;
          while (root->Parent() != NULL)
          {
            if (stillShrinking)
              stillShrinking = root->ShrinkBoundForBound(bound);
            root = root->Parent();
          }
          if (stillShrinking)
            root->ShrinkBoundForBound(bound);

          root = parent;
          while (root != NULL)
          {
            root->numDescendants -= numDescendants;
            root = root->Parent();
          }

          stillShrinking = true;
          root = parent;
          while (root->Parent() != NULL)
          {
            if (stillShrinking)
              stillShrinking = root->AuxiliaryInfo().UpdateAuxiliaryInfo(root);
            root = root->Parent();
          }
          if (stillShrinking)
            root->AuxiliaryInfo().UpdateAuxiliaryInfo(root);

          // Reinsert the nodes at the root node.
          for (size_t i = 0; i < numChildren; ++i)
            root->InsertNode(children[i], level, relevels);

          // This will check the minFill of the point.
          parent->CondenseTree(point, relevels, usePoint);
          // Now it should be safe to delete this node.
          SoftDelete();

          return;
        }
      }
    }
    else if (numChildren == 1)
    {
      // If there are multiple children, we can't do anything to the root.
      RectangleTree* child = children[0];

      // Required for the X tree.
      if (child->NumChildren() > maxNumChildren)
      {
        maxNumChildren = child->MaxNumChildren();
        children.resize(maxNumChildren + 1);
      }

      for (size_t i = 0; i < child->NumChildren(); ++i)
      {
        children[i] = child->children[i];
        children[i]->Parent() = this;
        child->children[i] = NULL;
      }

      numChildren = child->NumChildren();
      child->NumChildren() = 0;

      for (size_t i = 0; i < child->Count(); ++i)
      {
        // In case the tree has a height of two.
        points[i] = child->Point(i);
      }

      auxiliaryInfo = child->AuxiliaryInfo();

      count = child->Count();
      child->Count() = 0;

      delete child;
      return;
    }
  }

  // If we didn't delete it, shrink the bound if we need to.
  if (usePoint &&
      (ShrinkBoundForPoint(point) || auxiliaryInfo.UpdateAuxiliaryInfo(this)) &&
      parent != NULL)
    parent->CondenseTree(point, relevels, usePoint);
  else if (!usePoint &&
           (ShrinkBoundForBound(bound) ||
           auxiliaryInfo.UpdateAuxiliaryInfo(this)) &&
           parent != NULL)
    parent->CondenseTree(point, relevels, usePoint);
}

/**
 * Shrink the bound so it fits tightly after the removal of this point.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
bool RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    ShrinkBoundForPoint(const arma::vec& point)
{
  bool shrunk = false;
  if (IsLeaf())
  {
    for (size_t i = 0; i < bound.Dim(); ++i)
    {
      if (bound[i].Lo() == point[i])
      {
        ElemType min = std::numeric_limits<ElemType>::max();
        for (size_t j = 0; j < count; ++j)
        {
          if (dataset->col(points[j])[i] < min)
            min = dataset->col(points[j])[i];
        }

        if (bound[i].Lo() < min)
        {
          shrunk = true;
          bound[i].Lo() = min;
        }
        else if (min < bound[i].Lo())
        {
          assert(false); // We have a problem.
        }
      }
      else if (bound[i].Hi() == point[i])
      {
        ElemType max = std::numeric_limits<ElemType>::lowest();
        for (size_t j = 0; j < count; ++j)
        {
          if (dataset->col(points[j])[i] > max)
            max = dataset->col(points[j])[i];
        }

        if (bound[i].Hi() > max)
        {
          shrunk = true;
          bound[i].Hi() = max;
        }
        else if (max > bound[i].Hi())
        {
          assert(false); // We have a problem.
        }
      }
    }
  }
  else
  {
    for (size_t i = 0; i < bound.Dim(); ++i)
    {
      if (bound[i].Lo() == point[i])
      {
        ElemType min = std::numeric_limits<ElemType>::max();
        for (size_t j = 0; j < numChildren; ++j)
        {
          if (children[j]->Bound()[i].Lo() < min)
            min = children[j]->Bound()[i].Lo();
        }

        if (bound[i].Lo() < min)
        {
          shrunk = true;
          bound[i].Lo() = min;
        }
      }
      else if (bound[i].Hi() == point[i])
      {
        ElemType max = std::numeric_limits<ElemType>::lowest();
        for (size_t j = 0; j < numChildren; ++j)
        {
          if (children[j]->Bound()[i].Hi() > max)
            max = children[j]->Bound()[i].Hi();
        }

        if (bound[i].Hi() > max)
        {
          shrunk = true;
          bound[i].Hi() = max;
        }
      }
    }
  }

  return shrunk;
}

/**
 * Shrink the bound so it fits tightly after the removal of another bound.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
bool RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::
    ShrinkBoundForBound(const HRectBound<DistanceType>& /* b */)
{
  // Using the sum is safe since none of the dimensions can increase.
  ElemType sum = 0;

  // I think it may be faster to just recalculate the whole thing.
  for (size_t i = 0; i < bound.Dim(); ++i)
  {
    sum += bound[i].Width();
    bound[i].Lo() = std::numeric_limits<ElemType>::max();
    bound[i].Hi() = std::numeric_limits<ElemType>::lowest();
  }

  for (size_t i = 0; i < numChildren; ++i)
  {
    bound |= children[i]->Bound();
  }

  ElemType sum2 = 0;
  for (size_t i = 0; i < bound.Dim(); ++i)
    sum2 += bound[i].Width();

  return sum != sum2;
}

/**
 * Serialize the tree.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         typename SplitType,
         typename DescentType,
         template<typename> class AuxiliaryInformationType>
template<typename Archive>
void RectangleTree<DistanceType, StatisticType, MatType, SplitType, DescentType,
                   AuxiliaryInformationType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  // Clean up memory, if necessary.
  if (cereal::is_loading<Archive>())
  {
    for (size_t i = 0; i < numChildren; ++i)
      delete children[i];
    children.clear();

    if (ownsDataset && dataset)
      delete dataset;

    parent = NULL;
  }

  bool hasParent = (parent != NULL);

  ar(CEREAL_NVP(maxNumChildren));
  ar(CEREAL_NVP(minNumChildren));
  ar(CEREAL_NVP(numChildren));
  if (cereal::is_loading<Archive>())
    children.resize(maxNumChildren + 1);

  ar(CEREAL_NVP(begin));
  ar(CEREAL_NVP(count));
  ar(CEREAL_NVP(numDescendants));
  ar(CEREAL_NVP(maxLeafSize));
  ar(CEREAL_NVP(minLeafSize));
  ar(CEREAL_NVP(bound));
  ar(CEREAL_NVP(stat));
  ar(CEREAL_NVP(parentDistance));
  ar(CEREAL_NVP(hasParent));

  if (!hasParent)
  {
    MatType*& datasetTemp = const_cast<MatType*&>(dataset);
    ar(CEREAL_POINTER(datasetTemp));
  }

  ar(CEREAL_NVP(points));
  ar(CEREAL_NVP(auxiliaryInfo));

  // Since we may or may not be holding children, we need to serialize _only_
  // numChildren children.
  for (size_t i = 0; i < numChildren; ++i)
  {
    std::ostringstream oss;
    oss << "children" << i;
    ar(CEREAL_POINTER(children[i]));

    if (cereal::is_loading<Archive>())
      children[i]->parent = this;
  }
  for (size_t i = numChildren; i < maxNumChildren + 1; ++i)
  {
    children[i] = NULL;
  }

  // If we are the root, we need to restore the dataset pointer throughout
  if (!hasParent)
  {
    std::stack<RectangleTree*> stack;
    for (size_t i = 0; i < numChildren; ++i)
    {
      stack.push(children[i]);
    }
    while (!stack.empty())
    {
      RectangleTree* node = stack.top();
      stack.pop();
      node->dataset = dataset;
      for (size_t i = 0; i < node->numChildren; ++i)
      {
        stack.push(node->children[i]);
      }
    }
  }
}

} // namespace mlpack

#endif
