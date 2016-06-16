/**
 * @file rectangle_tree_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of generalized rectangle tree.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "rectangle_tree.hpp"

#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/log.hpp>

namespace mlpack {
namespace tree {

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
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
    maxLeafSize(maxLeafSize),
    minLeafSize(minLeafSize),
    bound(data.n_rows),
    parentDistance(0),
    dataset(new MatType(data)),
    ownsDataset(true),
    points(maxLeafSize + 1), // Add one to make splitting the node simpler.
    localDataset(new MatType(arma::zeros<MatType>(data.n_rows,
                                                  maxLeafSize + 1)))
{
  stat = StatisticType(*this);

  split = SplitType<RectangleTree>(this);

  // For now, just insert the points in order.
  RectangleTree* root = this;

  for (size_t i = firstDataIndex; i < data.n_cols; i++)
    root->InsertPoint(i);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
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
    maxLeafSize(maxLeafSize),
    minLeafSize(minLeafSize),
    bound(data.n_rows),
    parentDistance(0),
    dataset(new MatType(std::move(data))),
    ownsDataset(true),
    points(maxLeafSize + 1), // Add one to make splitting the node simpler.
    localDataset(new MatType(arma::zeros<MatType>(dataset->n_rows,
                                                  maxLeafSize + 1)))
{
  stat = StatisticType(*this);

  split = SplitType<RectangleTree>(this);

  // For now, just insert the points in order.
  RectangleTree* root = this;

  for (size_t i = firstDataIndex; i < dataset->n_cols; i++)
    root->InsertPoint(i);
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
RectangleTree(
    RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>*
        parentNode, const size_t numMaxChildren) :
    maxNumChildren(numMaxChildren > 0 ? numMaxChildren :
        parentNode->MaxNumChildren()),
    minNumChildren(parentNode->MinNumChildren()),
    numChildren(0),
    children(maxNumChildren + 1),
    parent(parentNode),
    begin(0),
    count(0),
    maxLeafSize(parentNode->MaxLeafSize()),
    minLeafSize(parentNode->MinLeafSize()),
    bound(parentNode->Bound().Dim()),
    parentDistance(0),
    dataset(&parentNode->Dataset()),
    ownsDataset(false),
    points(maxLeafSize + 1), // Add one to make splitting the node simpler.
    localDataset(new MatType(arma::zeros<MatType>(parentNode->Bound().Dim(),
                                                  maxLeafSize + 1)))
{
  stat = StatisticType(*this);
  split = SplitType<RectangleTree>(this);
}

/**
 * Create a rectangle tree by copying the other tree.  Be careful!  This can
 * take a long time and use a lot of memory.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
RectangleTree(
    const RectangleTree& other,
    const bool deepCopy) :
    maxNumChildren(other.MaxNumChildren()),
    minNumChildren(other.MinNumChildren()),
    numChildren(other.NumChildren()),
    children(maxNumChildren + 1),
    parent(other.Parent()),
    begin(other.Begin()),
    count(other.Count()),
    maxLeafSize(other.MaxLeafSize()),
    minLeafSize(other.MinLeafSize()),
    bound(other.bound),
    parentDistance(other.ParentDistance()),
    dataset(deepCopy ? new MatType(*other.dataset) : &other.Dataset()),
    ownsDataset(deepCopy),
    points(other.Points()),
    localDataset(NULL)
{
  split = SplitType<RectangleTree>(other);
  if (deepCopy)
  {
    if (numChildren > 0)
    {
      for (size_t i = 0; i < numChildren; i++)
      {
        children[i] = new RectangleTree(*(other.Children()[i]));
      }
    }
    else
    {
      localDataset = new MatType(other.LocalDataset());
    }
  }
  else
  {
    children = other.Children();
    arma::mat& otherData = const_cast<arma::mat&>(other.LocalDataset());
    localDataset = &otherData;
  }
}

/**
 * Construct the tree from a boost::serialization archive.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
template<typename Archive>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
RectangleTree(
    Archive& ar,
    const typename boost::enable_if<typename Archive::is_loading>::type*) :
    RectangleTree() // Use default constructor.
{
  // Now serialize.
  ar >> data::CreateNVP(*this, "tree");
}

/**
 * Deletes this node, deallocating the memory for the children and calling
 * their destructors in turn.  This will invalidate any pointers or references
 * to any nodes which are children of this one.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
~RectangleTree()
{
  for (size_t i = 0; i < numChildren; i++)
    delete children[i];

  if (ownsDataset)
    delete dataset;

  delete localDataset;
}

/**
 * Deletes this node but leaves the children untouched.  Needed for when we
 * split nodes and remove nodes (inserting and deleting points).
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    SoftDelete()
{
  parent = NULL;

  for (size_t i = 0; i < children.size(); i++)
    children[i] = NULL;

  numChildren = 0;
  delete this;
}

/**
 * Set the local dataset to null.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    NullifyData()
{
  localDataset = NULL;
}

/**
 * Recurse through the tree and insert the point at the leaf node chosen
 * by the heuristic.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    InsertPoint(const size_t point)
{
  // Expand the bound regardless of whether it is a leaf node.
  bound |= dataset->col(point);

  std::vector<bool> lvls(TreeDepth());
  for (size_t i = 0; i < lvls.size(); i++)
    lvls[i] = true;

  // If this is a leaf node, we stop here and add the point.
  if (numChildren == 0)
  {
    localDataset->col(count) = dataset->col(point);
    points[count++] = point;
    SplitNode(lvls);
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  const size_t descentNode = DescentType::ChooseDescentNode(this,
      dataset->col(point));
  children[descentNode]->InsertPoint(point, lvls);
}

/**
 * Inserts a point into the tree, tracking which levels have been inserted into.
 * The point will be copied to the data matrix of the leaf node where it is
 * finally inserted, but we pass by reference since it may be passed many times
 * before it actually reaches a leaf.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    InsertPoint(const size_t point, std::vector<bool>& relevels)
{
  // Expand the bound regardless of whether it is a leaf node.
  bound |= dataset->col(point);

  // If this is a leaf node, we stop here and add the point.
  if (numChildren == 0)
  {
    localDataset->col(count) = dataset->col(point);
    points[count++] = point;
    SplitNode(relevels);
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  const size_t descentNode = DescentType::ChooseDescentNode(this,
      dataset->col(point));
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
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    InsertNode(RectangleTree* node,
               const size_t level,
               std::vector<bool>& relevels)
{
  // Expand the bound regardless of the level.
  bound |= node->Bound();
  if (level == TreeDepth())
  {
    children[numChildren++] = node;
    node->Parent() = this;
    SplitNode(relevels);
  }
  else
  {
    const size_t descentNode = DescentType::ChooseDescentNode(this, node);
    children[descentNode]->InsertNode(node, level, relevels);
  }
}

/**
 * Recurse through the tree to remove the point.  Once we find the point, we
 * shrink the rectangles if necessary.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
bool RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    DeletePoint(const size_t point)
{
  // It is possible that this will cause a reinsertion, so we need to handle the
  // levels properly.
  RectangleTree* root = this;
  while (root->Parent() != NULL)
    root = root->Parent();

  std::vector<bool> lvls(root->TreeDepth());
  for (size_t i = 0; i < lvls.size(); i++)
    lvls[i] = true;

  if (numChildren == 0)
  {
    for (size_t i = 0; i < count; i++)
    {
      if (points[i] == point)
      {
        localDataset->col(i) = localDataset->col(--count); // Decrement count.
        points[i] = points[count];
        // This function wil ensure that minFill is satisfied.
        CondenseTree(dataset->col(point), lvls, true);
        return true;
      }
    }
  }

  for (size_t i = 0; i < numChildren; i++)
    if (children[i]->Bound().Contains(dataset->col(point)))
      if (children[i]->DeletePoint(point, lvls))
        return true;

  return false;
}

/**
 * Recurse through the tree to remove the point.  Once we find the point, we
 * shrink the rectangles if necessary.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
bool RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    DeletePoint(const size_t point, std::vector<bool>& relevels)
{
  if (numChildren == 0)
  {
    for (size_t i = 0; i < count; i++)
    {
      if (points[i] == point)
      {
        localDataset->col(i) = localDataset->col(--count);
        points[i] = points[count];
        // This function will ensure that minFill is satisfied.
        CondenseTree(dataset->col(point), relevels, true);
        return true;
      }
    }
  }

  for (size_t i = 0; i < numChildren; i++)
    if (children[i]->Bound().Contains(dataset->col(point)))
      if (children[i]->DeletePoint(point, relevels))
        return true;

  return false;
}

/**
 * Recurse through the tree to remove the node.  Once we find the node, we
 * shrink the rectangles if necessary.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
bool RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    RemoveNode(const RectangleTree* node, std::vector<bool>& relevels)
{
  for (size_t i = 0; i < numChildren; i++)
  {
    if (children[i] == node)
    {
      children[i] = children[--numChildren]; // Decrement numChildren.
      CondenseTree(arma::vec(), relevels, false);
      return true;
    }

    bool contains = true;
    for (size_t j = 0; j < node->Bound().Dim(); j++)
      contains &= Children()[i]->Bound()[j].Contains(node->Bound()[j]);

    if (contains)
      if (children[i]->RemoveNode(node, relevels))
        return true;
  }

  return false;
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
size_t RectangleTree<MetricType, StatisticType, MatType, SplitType,
                     DescentType>::TreeSize() const
{
  int n = 0;
  for (int i = 0; i < numChildren; i++)
    n += children[i]->TreeSize();

  return n + 1; // Add one for this node.
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
size_t RectangleTree<MetricType, StatisticType, MatType, SplitType,
                     DescentType>::TreeDepth() const
{
  int n = 1;
  RectangleTree* currentNode = const_cast<RectangleTree*> (this);

  while (!currentNode->IsLeaf())
  {
    currentNode = currentNode->Children()[0];
    n++;
  }

  return n;
}

template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
inline bool RectangleTree<MetricType, StatisticType, MatType, SplitType,
                          DescentType>::IsLeaf() const
{
  return (numChildren == 0);
}

/**
 * Return a bound on the furthest point in the node form the centroid.
 * This returns 0 unless the node is a leaf.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
inline
typename RectangleTree<MetricType, StatisticType, MatType, SplitType,
    DescentType>::ElemType
RectangleTree<MetricType, StatisticType, MatType, SplitType,
    DescentType>::FurthestPointDistance() const
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
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
inline
typename RectangleTree<MetricType, StatisticType, MatType, SplitType,
    DescentType>::ElemType
RectangleTree<MetricType, StatisticType, MatType, SplitType,
    DescentType>::FurthestDescendantDistance() const
{
  // Return the distance from the centroid to a corner of the bound.
  return 0.5 * bound.Diameter();
}

/**
 * Return the number of points contained in this node.  Zero if it is a non-leaf
 * node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
inline size_t RectangleTree<MetricType, StatisticType, MatType, SplitType,
                            DescentType>::NumPoints() const
{
  if (numChildren != 0) // This is not a leaf node.
    return 0;

  return count;
}

/**
 * Return the number of descendants under or in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
inline size_t RectangleTree<MetricType, StatisticType, MatType, SplitType,
                            DescentType>::NumDescendants() const
{
  if (numChildren == 0)
  {
    return count;
  }
  else
  {
    size_t n = 0;
    for (size_t i = 0; i < numChildren; i++)
      n += children[i]->NumDescendants();
    return n;
  }
}

/**
 * Return the index of a particular descendant contained in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
inline size_t RectangleTree<MetricType, StatisticType, MatType, SplitType,
                            DescentType>::Descendant(const size_t index) const
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
 * Return the index of a particular point contained in this node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
inline size_t RectangleTree<MetricType, StatisticType, MatType, SplitType,
                            DescentType>::Point(const size_t index) const
{
  return points[index];
}

/**
 * Split the tree.  This calls the SplitType code to split a node.  This method
 * should only be called on a leaf node.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    SplitNode(std::vector<bool>& relevels)
{
  if (numChildren == 0)
  {
    // Check to see if we are full.
    if (count <= maxLeafSize)
      return; // We don't need to split.

    // If we are full, then we need to split (or at least try).  The SplitType
    // takes care of this and of moving up the tree if necessary.
    split.SplitLeafNode(this, relevels);
  }
  else
  {
    // Check to see if we are full.
    if (numChildren <= maxNumChildren)
      return; // We don't need to split.

    // If we are full, then we need to split (or at least try).  The SplitType
    // takes care of this and of moving up the tree if necessary.
    split.SplitNonLeafNode(this, relevels);
  }
}

//! Default constructor for boost::serialization.
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
RectangleTree() :
    maxNumChildren(0), // Try to give sensible defaults, but it shouldn't matter
    minNumChildren(0), // because this tree isn't valid anyway and is only used
    numChildren(0),    // by boost::serialization.
    parent(NULL),
    begin(0),
    count(0),
    maxLeafSize(0),
    minLeafSize(0),
    parentDistance(0.0),
    dataset(NULL),
    ownsDataset(false),
    localDataset(NULL)
{
  // Nothing to do.
}

/**
 * Condense the tree.  This shrinks the bounds and moves up the tree if
 * applicable.  If a node goes below minimum fill, this code will deal with it.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    CondenseTree(const arma::vec& point,
                 std::vector<bool>& relevels,
                 const bool usePoint)
{
  // First delete the node if we need to.  There's no point in shrinking the
  // bound first.
  if (IsLeaf() && count < minLeafSize && parent != NULL)
  {
    // We can't delete the root node.
    for (size_t i = 0; i < parent->NumChildren(); i++)
    {
      if (parent->Children()[i] == this)
      {
        // Decrement numChildren.
        parent->Children()[i] = parent->Children()[--parent->NumChildren()];

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
          stillShrinking = root->ShrinkBoundForBound(bound);

        // Reinsert the points at the root node.
        for (size_t j = 0; j < count; j++)
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
      for (size_t j = 0; j < parent->NumChildren(); j++)
      {
        if (parent->Children()[j] == this)
        {
          // Decrement numChildren.
          parent->Children()[j] = parent->Children()[--parent->NumChildren()];
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
            stillShrinking = root->ShrinkBoundForBound(bound);

          // Reinsert the nodes at the root node.
          for (size_t i = 0; i < numChildren; i++)
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
      if(child->NumChildren() > maxNumChildren)
      {
        maxNumChildren = child->MaxNumChildren();
        children.resize(maxNumChildren+1);
      }

      for (size_t i = 0; i < child->NumChildren(); i++) {
        children[i] = child->Children()[i];
        children[i]->Parent() = this;
      }

      numChildren = child->NumChildren();

      for (size_t i = 0; i < child->Count(); i++)
      {
        // In case the tree has a height of two.
        points[i] = child->Points()[i];
        localDataset->col(i) = child->LocalDataset().col(i);
      }

      count = child->Count();
      child->SoftDelete();
      return;
    }
  }

  // If we didn't delete it, shrink the bound if we need to.
  if (usePoint && ShrinkBoundForPoint(point) && parent != NULL)
    parent->CondenseTree(point, relevels, usePoint);
  else if (!usePoint && ShrinkBoundForBound(bound) && parent != NULL)
    parent->CondenseTree(point, relevels, usePoint);
}

/**
 * Shrink the bound so it fits tightly after the removal of this point.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
bool RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    ShrinkBoundForPoint(const arma::vec& point)
{
  bool shrunk = false;
  if (IsLeaf())
  {
    for (size_t i = 0; i < bound.Dim(); i++)
    {
      if (bound[i].Lo() == point[i])
      {
        ElemType min = std::numeric_limits<ElemType>::max();
        for (size_t j = 0; j < count; j++)
        {
          if (localDataset->col(j)[i] < min)
            min = localDataset->col(j)[i];
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
        for (size_t j = 0; j < count; j++)
        {
          if (localDataset->col(j)[i] > max)
            max = localDataset->col(j)[i];
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
    for (size_t i = 0; i < bound.Dim(); i++)
    {
      if (bound[i].Lo() == point[i])
      {
        ElemType min = std::numeric_limits<ElemType>::max();
        for (size_t j = 0; j < numChildren; j++)
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
        for (size_t j = 0; j < numChildren; j++)
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
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
bool RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    ShrinkBoundForBound(const bound::HRectBound<MetricType>& /* b */)
{
  // Using the sum is safe since none of the dimensions can increase.
  ElemType sum = 0;

  // I think it may be faster to just recalculate the whole thing.
  for (size_t i = 0; i < bound.Dim(); i++)
  {
    sum += bound[i].Width();
    bound[i].Lo() = std::numeric_limits<ElemType>::max();
    bound[i].Hi() = std::numeric_limits<ElemType>::lowest();
  }

  for (size_t i = 0; i < numChildren; i++)
  {
    bound |= children[i]->Bound();
  }

  ElemType sum2 = 0;
  for (size_t i = 0; i < bound.Dim(); i++)
    sum2 += bound[i].Width();

  return sum != sum2;
}

/**
 * Serialize the tree.
 */
template<typename MetricType,
         typename StatisticType,
         typename MatType,
         template<typename> class SplitType,
         typename DescentType>
template<typename Archive>
void RectangleTree<MetricType, StatisticType, MatType, SplitType, DescentType>::
    Serialize(Archive& ar,
              const unsigned int /* version */)
{
  using data::CreateNVP;

  // Clean up memory, if necessary.
  if (Archive::is_loading::value)
  {
    for (size_t i = 0; i < numChildren; i++)
      delete children[i];
    children.clear();

    if (ownsDataset && dataset)
      delete dataset;

    if (localDataset)
      delete localDataset;
  }

  ar & CreateNVP(maxNumChildren, "maxNumChildren");
  ar & CreateNVP(minNumChildren, "minNumChildren");
  ar & CreateNVP(numChildren, "numChildren");

  // Due to quirks of boost::serialization, depending on how the user serializes
  // the tree, the root node may be duplicated.  Therefore we don't allow
  // children of the root to serialize the parent, and we fix the parent link
  // after serializing the children when loading below.
  if (Archive::is_saving::value && parent != NULL && parent->Parent() == NULL)
  {
    RectangleTree* fakeParent = NULL;
    ar & CreateNVP(fakeParent, "parent");
  }
  else
  {
    ar & CreateNVP(parent, "parent");
  }

  ar & CreateNVP(begin, "begin");
  ar & CreateNVP(count, "count");
  ar & CreateNVP(maxLeafSize, "maxLeafSize");
  ar & CreateNVP(minLeafSize, "minLeafSize");
  ar & CreateNVP(bound, "bound");
  ar & CreateNVP(stat, "stat");
  ar & CreateNVP(parentDistance, "parentDistance");
  ar & CreateNVP(dataset, "dataset");

  // If we are loading and we are the root, we own the dataset.
  if (Archive::is_loading::value && parent == NULL)
    ownsDataset = true;

  ar & CreateNVP(points, "points");
  ar & CreateNVP(localDataset, "localDataset");
  ar & CreateNVP(split, "split");

  // Because 'children' holds mlpack types (that have Serialize()), we can't use
  // the std::vector serialization.
  if (Archive::is_loading::value)
    children.resize(numChildren);
  for (size_t i = 0; i < numChildren; ++i)
  {
    std::ostringstream oss;
    oss << "child" << i;
    ar & CreateNVP(children[i], oss.str());
  }

  // Fix the parent links for the children, if necessary.
  if (Archive::is_loading::value && parent == NULL)
  {
    // Look through each child individually.
    for (size_t i = 0; i < children.size(); ++i)
    {
      children[i]->ownsDataset = false;
      children[i]->Parent() = this;
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif
