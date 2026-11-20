/**
 * @file core/tree/binary_space_tree/binary_space_tree_impl.hpp
 *
 * Implementation of generalized space partitioning tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "binary_space_tree.hpp"

#include <mlpack/core/util/log.hpp>
#include <queue>

namespace mlpack {

// Default constructor.
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
    BinarySpaceTree() :
    left(NULL),
    right(NULL),
    parent(NULL),
    begin(0),
    count(0),
    stat(*this),
    parentDistance(0),
    furthestDescendantDistance(0),
    dataset(NULL)
{
  // Nothing to do.
}

// Each of these overloads is kept as a separate function to keep the overhead
// from the two std::vectors out, if possible.
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    const MatType& data,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(NULL),
    begin(0), /* This root node starts at index 0, */
    count(data.n_cols), /* and spans all of the dataset. */
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(data)) // Copies the dataset.
{
  // Do the actual splitting of this node.
  SplitType<BoundType<DistanceType, ElemType>, MatType> splitter;
  SplitNode(maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    const MatType& data,
    std::vector<size_t>& oldFromNew,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(NULL),
    begin(0),
    count(data.n_cols),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(data)) // Copies the dataset.
{
  // Initialize oldFromNew correctly.
  oldFromNew.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    oldFromNew[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitType<BoundType<DistanceType, ElemType>, MatType> splitter;
  SplitNode(oldFromNew, maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    const MatType& data,
    std::vector<size_t>& oldFromNew,
    std::vector<size_t>& newFromOld,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(NULL),
    begin(0),
    count(data.n_cols),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(data)) // Copies the dataset.
{
  // Initialize the oldFromNew vector correctly.
  oldFromNew.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    oldFromNew[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitType<BoundType<DistanceType, ElemType>, MatType> splitter;
  SplitNode(oldFromNew, maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);

  // Map the newFromOld indices correctly.
  newFromOld.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    newFromOld[oldFromNew[i]] = i;
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(MatType&& data, const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(NULL),
    begin(0),
    count(data.n_cols),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(std::move(data)))
{
  // Do the actual splitting of this node.
  SplitType<BoundType<DistanceType, ElemType>, MatType> splitter;
  SplitNode(maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    MatType&& data,
    std::vector<size_t>& oldFromNew,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(NULL),
    begin(0),
    count(data.n_cols),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(std::move(data)))
{
  // Initialize oldFromNew correctly.
  oldFromNew.resize(dataset->n_cols);
  for (size_t i = 0; i < dataset->n_cols; ++i)
    oldFromNew[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitType<BoundType<DistanceType, ElemType>, MatType> splitter;
  SplitNode(oldFromNew, maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    MatType&& data,
    std::vector<size_t>& oldFromNew,
    std::vector<size_t>& newFromOld,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(NULL),
    begin(0),
    count(data.n_cols),
    bound(data.n_rows),
    parentDistance(0), // Parent distance for the root is 0: it has no parent.
    dataset(new MatType(std::move(data)))
{
  // Initialize the oldFromNew vector correctly.
  oldFromNew.resize(dataset->n_cols);
  for (size_t i = 0; i < dataset->n_cols; ++i)
    oldFromNew[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitType<BoundType<DistanceType, ElemType>, MatType> splitter;
  SplitNode(oldFromNew, maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);

  // Map the newFromOld indices correctly.
  newFromOld.resize(dataset->n_cols);
  for (size_t i = 0; i < dataset->n_cols; ++i)
    newFromOld[oldFromNew[i]] = i;
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    BinarySpaceTree* parent,
    const size_t begin,
    const size_t count,
    SplitType<BoundType<DistanceType, ElemType>, MatType>& splitter,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(parent),
    begin(begin),
    count(count),
    bound(parent->Dataset().n_rows),
    dataset(&parent->Dataset()) // Point to the parent's dataset.
{
  // Perform the actual splitting.
  SplitNode(maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    BinarySpaceTree* parent,
    const size_t begin,
    const size_t count,
    std::vector<size_t>& oldFromNew,
    SplitType<BoundType<DistanceType, ElemType>, MatType>& splitter,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(parent),
    begin(begin),
    count(count),
    bound(parent->Dataset().n_rows),
    dataset(&parent->Dataset())
{
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  assert(oldFromNew.size() == dataset->n_cols);

  // Perform the actual splitting.
  SplitNode(oldFromNew, maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    BinarySpaceTree* parent,
    const size_t begin,
    const size_t count,
    std::vector<size_t>& oldFromNew,
    std::vector<size_t>& newFromOld,
    SplitType<BoundType<DistanceType, ElemType>, MatType>& splitter,
    const size_t maxLeafSize) :
    left(NULL),
    right(NULL),
    parent(parent),
    begin(begin),
    count(count),
    bound(parent->Dataset()->n_rows),
    dataset(&parent->Dataset())
{
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  Log::Assert(oldFromNew.size() == dataset->n_cols);

  // Perform the actual splitting.
  SplitNode(oldFromNew, maxLeafSize, splitter);

  // Create the statistic depending on if we are a leaf or not.
  stat = StatisticType(*this);

  // Map the newFromOld indices correctly.
  newFromOld.resize(dataset->n_cols);
  for (size_t i = 0; i < dataset->n_cols; ++i)
    newFromOld[oldFromNew[i]] = i;
}

/**
 * Create a binary space tree by copying the other tree.  Be careful!  This can
 * take a long time and use a lot of memory.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    const BinarySpaceTree& other) :
    left(NULL),
    right(NULL),
    parent(other.parent),
    begin(other.begin),
    count(other.count),
    bound(other.bound),
    stat(other.stat),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    minimumBoundDistance(other.minimumBoundDistance),
    // Copy matrix, but only if we are the root.
    dataset((other.parent == NULL) ? new MatType(*other.dataset) : NULL)
{
  // Create left and right children (if any).
  if (other.Left())
  {
    left = new BinarySpaceTree(*other.Left());
    left->Parent() = this; // Set parent to this, not other tree.
  }

  if (other.Right())
  {
    right = new BinarySpaceTree(*other.Right());
    right->Parent() = this; // Set parent to this, not other tree.
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL)
  {
    std::queue<BinarySpaceTree*> queue;
    if (left)
      queue.push(left);
    if (right)
      queue.push(right);
    while (!queue.empty())
    {
      BinarySpaceTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      if (node->left)
        queue.push(node->left);
      if (node->right)
        queue.push(node->right);
    }
  }
}

/**
 * Copy assignment operator: copy the given other tree.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>&
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
operator=(const BinarySpaceTree& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  delete dataset;
  delete left;
  delete right;

  left = NULL;
  right = NULL;
  parent = other.Parent();
  begin = other.Begin();
  count = other.Count();
  bound = other.bound;
  stat = other.stat;
  parentDistance = other.ParentDistance();
  furthestDescendantDistance = other.FurthestDescendantDistance();
  minimumBoundDistance = other.MinimumBoundDistance();
  // Copy matrix, but only if we are the root.
  dataset = ((other.parent == NULL) ? new MatType(*other.dataset) : NULL);

  // Create left and right children (if any).
  if (other.Left())
  {
    left = new BinarySpaceTree(*other.Left());
    left->Parent() = this; // Set parent to this, not other tree.
  }

  if (other.Right())
  {
    right = new BinarySpaceTree(*other.Right());
    right->Parent() = this; // Set parent to this, not other tree.
  }

  // Propagate matrix, but only if we are the root.
  if (parent == NULL)
  {
    std::queue<BinarySpaceTree*> queue;
    if (left)
      queue.push(left);
    if (right)
      queue.push(right);
    while (!queue.empty())
    {
      BinarySpaceTree* node = queue.front();
      queue.pop();

      node->dataset = dataset;
      if (node->left)
        queue.push(node->left);
      if (node->right)
        queue.push(node->right);
    }
  }

  return *this;
}

/**
 * Move assignment operator: take ownership of the given tree.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>&
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
operator=(BinarySpaceTree&& other)
{
  // Return if it's the same tree.
  if (this == &other)
    return *this;

  // Freeing memory that will not be used anymore.
  delete dataset;
  delete left;
  delete right;

  parent = other.Parent();
  left = other.Left();
  right = other.Right();
  begin = other.Begin();
  count = other.Count();
  bound = std::move(other.bound);
  stat = std::move(other.stat);
  parentDistance = other.ParentDistance();
  furthestDescendantDistance = other.FurthestDescendantDistance();
  minimumBoundDistance = other.MinimumBoundDistance();
  dataset = other.dataset;

  other.left = NULL;
  other.right = NULL;
  other.parent = NULL;
  other.begin = 0;
  other.count = 0;
  other.parentDistance = 0.0;
  other.furthestDescendantDistance = 0.0;
  other.minimumBoundDistance = 0.0;
  other.dataset = NULL;

  return *this;
}


/**
 * Move constructor.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(BinarySpaceTree&& other) :
    left(other.left),
    right(other.right),
    parent(other.parent),
    begin(other.begin),
    count(other.count),
    bound(std::move(other.bound)),
    stat(std::move(other.stat)),
    parentDistance(other.parentDistance),
    furthestDescendantDistance(other.furthestDescendantDistance),
    minimumBoundDistance(other.minimumBoundDistance),
    dataset(other.dataset)
{
  // Now we are a clone of the other tree.  But we must also clear the other
  // tree's contents, so it doesn't delete anything when it is destructed.
  other.left = NULL;
  other.right = NULL;
  other.parent = NULL;
  other.begin = 0;
  other.count = 0;
  other.parentDistance = 0.0;
  other.furthestDescendantDistance = 0.0;
  other.minimumBoundDistance = 0.0;
  other.dataset = NULL;

  // Set new parent.
  if (left)
    left->parent = this;
  if (right)
    right->parent = this;
}

/**
 * Initialize the tree from an archive.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
template<typename Archive>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
BinarySpaceTree(
    Archive& ar,
    const typename std::enable_if_t<cereal::is_loading<Archive>()>*) :
    BinarySpaceTree() // Create an empty BinarySpaceTree.
{
  // We've delegated to the constructor which gives us an empty tree, and now we
  // can serialize from it.
  ar(CEREAL_NVP(*this));
}

/**
 * Deletes this node, deallocating the memory for the children and calling their
 * destructors in turn.  This will invalidate any pointers or references to any
 * nodes which are children of this one.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
    ~BinarySpaceTree()
{
  delete left;
  delete right;

  // If we're the root, delete the matrix.
  if (!parent)
    delete dataset;
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline bool BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                            SplitType>::IsLeaf() const
{
  return !left;
}

/**
 * Returns the number of children in this node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                              SplitType>::NumChildren() const
{
  if (left && right)
    return 2;
  if (left)
    return 1;

  return 0;
}

/**
 * Return the index of the nearest child node to the given query point.  If
 * this is a leaf node, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
template<typename VecType>
size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::GetNearestChild(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf() || !left || !right)
    return 0;

  if (left->MinDistance(point) <= right->MinDistance(point))
    return 0;
  return 1;
}

/**
 * Return the index of the furthest child node to the given query point.  If
 * this is a leaf node, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
template<typename VecType>
size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::GetFurthestChild(
    const VecType& point,
    typename std::enable_if_t<IsVector<VecType>::value>*)
{
  if (IsLeaf() || !left || !right)
    return 0;

  if (left->MaxDistance(point) > right->MaxDistance(point))
    return 0;
  return 1;
}

/**
 * Return the index of the nearest child node to the given query node.  If it
 * can't decide, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::GetNearestChild(const BinarySpaceTree& queryNode)
{
  if (IsLeaf() || !left || !right)
    return 0;

  ElemType leftDist = left->MinDistance(queryNode);
  ElemType rightDist = right->MinDistance(queryNode);
  if (leftDist < rightDist)
    return 0;
  if (rightDist < leftDist)
    return 1;
  return NumChildren();
}

/**
 * Return the index of the furthest child node to the given query node.  If it
 * can't decide, it will return NumChildren() (invalid index).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::GetFurthestChild(const BinarySpaceTree& queryNode)
{
  if (IsLeaf() || !left || !right)
    return 0;

  ElemType leftDist = left->MaxDistance(queryNode);
  ElemType rightDist = right->MaxDistance(queryNode);
  if (leftDist > rightDist)
    return 0;
  if (rightDist > leftDist)
    return 1;
  return NumChildren();
}

/**
 * Return a bound on the furthest point in the node from the center.  This
 * returns 0 unless the node is a leaf.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline
typename BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::ElemType
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::FurthestPointDistance() const
{
  if (!IsLeaf())
    return 0.0;

  // Otherwise return the distance from the center to a corner of the bound.
  return 0.5 * bound.Diameter();
}

/**
 * Return the furthest possible descendant distance.  This returns the maximum
 * distance from the center to the edge of the bound and not the empirical
 * quantity which is the actual furthest descendant distance.  So the actual
 * furthest descendant distance may be less than what this method returns (but
 * it will never be greater than this).
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline
typename BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::ElemType
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::FurthestDescendantDistance() const
{
  return furthestDescendantDistance;
}

//! Return the minimum distance from the center to any bound edge.
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline
typename BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::ElemType
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
    SplitType>::MinimumBoundDistance() const
{
  return bound.MinWidth() / 2.0;
}

/**
 * Return the specified child.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                       SplitType>&
    BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                    SplitType>::Child(const size_t child) const
{
  if (child == 0)
    return *left;
  else
    return *right;
}

/**
 * Return the number of points contained in this node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                              SplitType>::NumPoints() const
{
  if (left)
    return 0;

  return count;
}

/**
 * Return the number of descendants contained in the node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                              SplitType>::NumDescendants() const
{
  return count;
}

/**
 * Return the index of a particular descendant contained in this node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                              SplitType>::Descendant(const size_t index) const
{
  return (begin + index);
}

/**
 * Return the index of a particular point contained in this node.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
inline size_t BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType,
                              SplitType>::Point(const size_t index) const
{
  return (begin + index);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
void
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
    SplitNode(const size_t maxLeafSize,
              SplitType<BoundType<DistanceType, ElemType>, MatType>& splitter)
{
  // We need to expand the bounds of this node properly.
  UpdateBound(bound);

  // Calculate the furthest descendant distance.
  furthestDescendantDistance = 0.5 * bound.Diameter();

  // Now, check if we need to split at all.
  if (count <= maxLeafSize)
    return; // We can't split this.

  // splitCol denotes the two partitions of the dataset after the split. The
  // points on its left go to the left child and the others go to the right
  // child.
  size_t splitCol;

  // Find the partition of the node. This method does not perform the split.
  typename Split::SplitInfo splitInfo;

  const bool split = splitter.SplitNode(bound, *dataset, begin, count,
      splitInfo);

  // The node may not be always split. For instance, if all the points are the
  // same, we can't split them.
  if (!split)
    return;

  // Perform the actual splitting.  This will order the dataset such that
  // points that belong to the left subtree are on the left of splitCol, and
  // points from the right subtree are on the right side of splitCol.
  splitCol = splitter.PerformSplit(*dataset, begin, count, splitInfo);

  assert(splitCol > begin);
  assert(splitCol < begin + count);

  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left = new BinarySpaceTree(this, begin, splitCol - begin, splitter,
      maxLeafSize);
  right = new BinarySpaceTree(this, splitCol, begin + count - splitCol,
      splitter, maxLeafSize);

  // Calculate parent distances for those two nodes.
  arma::Col<ElemType> center, leftCenter, rightCenter;
  Center(center);
  left->Center(leftCenter);
  right->Center(rightCenter);

  const ElemType leftParentDistance = bound.Distance().Evaluate(center,
      leftCenter);
  const ElemType rightParentDistance = bound.Distance().Evaluate(center,
      rightCenter);

  left->ParentDistance() = leftParentDistance;
  right->ParentDistance() = rightParentDistance;
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
void
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
SplitNode(std::vector<size_t>& oldFromNew,
          const size_t maxLeafSize,
          SplitType<BoundType<DistanceType, ElemType>, MatType>& splitter)
{
  // We need to expand the bounds of this node properly.
  UpdateBound(bound);

  // Calculate the furthest descendant distance.
  furthestDescendantDistance = 0.5 * bound.Diameter();

  // First, check if we need to split at all.
  if (count <= maxLeafSize)
    return; // We can't split this.

  // splitCol denotes the two partitions of the dataset after the split. The
  // points on its left go to the left child and the others go to the right
  // child.
  size_t splitCol;

  // Find the partition of the node. This method does not perform the split.
  typename Split::SplitInfo splitInfo;

  const bool split = splitter.SplitNode(bound, *dataset, begin, count,
      splitInfo);

  // The node may not be always split. For instance, if all the points are the
  // same, we can't split them.
  if (!split)
    return;

  // Perform the actual splitting.  This will order the dataset such that
  // points that belong to the left subtree are on the left of splitCol, and
  // points from the right subtree are on the right side of splitCol.
  splitCol = splitter.PerformSplit(*dataset, begin, count, splitInfo,
      oldFromNew);

  assert(splitCol > begin);
  assert(splitCol < begin + count);

  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left = new BinarySpaceTree(this, begin, splitCol - begin, oldFromNew,
      splitter, maxLeafSize);
  right = new BinarySpaceTree(this, splitCol, begin + count - splitCol,
      oldFromNew, splitter, maxLeafSize);

  // Calculate parent distances for those two nodes.
  arma::Col<ElemType> center, leftCenter, rightCenter;
  Center(center);
  left->Center(leftCenter);
  right->Center(rightCenter);

  const ElemType leftParentDistance = bound.Distance().Evaluate(center,
      leftCenter);
  const ElemType rightParentDistance = bound.Distance().Evaluate(center,
      rightCenter);

  left->ParentDistance() = leftParentDistance;
  right->ParentDistance() = rightParentDistance;
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
template<typename BoundType2>
void
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
UpdateBound(BoundType2& boundToUpdate)
{
  if (count > 0)
    boundToUpdate |= dataset->cols(begin, begin + count - 1);
}

template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
void
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
UpdateBound(HollowBallBound<DistanceType, ElemType>& boundToUpdate)
{
  if (!parent)
  {
    if (count > 0)
      boundToUpdate |= dataset->cols(begin, begin + count - 1);
    return;
  }

  if (parent->left != NULL && parent->left != this)
  {
    boundToUpdate.HollowCenter() = parent->left->bound.Center();
    boundToUpdate.InnerRadius() = std::numeric_limits<ElemType>::max();
  }

  if (count > 0)
    boundToUpdate |= dataset->cols(begin, begin + count - 1);
}

/**
 * Serialize the tree.
 */
template<typename DistanceType,
         typename StatisticType,
         typename MatType,
         template<typename BoundDistanceType,
                  typename BoundElemType,
                  typename...> class BoundType,
         template<typename SplitBoundType,
                  typename SplitMatType> class SplitType>
template<typename Archive>
void
BinarySpaceTree<DistanceType, StatisticType, MatType, BoundType, SplitType>::
    serialize(Archive& ar, const uint32_t /* version */)
{
  // If we're loading, and we have children, they need to be deleted.
  if (cereal::is_loading<Archive>())
  {
    if (left)
      delete left;
    if (right)
      delete right;
    if (!parent)
      delete dataset;

    parent = NULL;
    left = NULL;
    right = NULL;
  }

  ar(CEREAL_NVP(begin));
  ar(CEREAL_NVP(count));
  ar(CEREAL_NVP(bound));
  ar(CEREAL_NVP(stat));

  ar(CEREAL_NVP(parentDistance));
  ar(CEREAL_NVP(furthestDescendantDistance));

  // Save children last.
  bool hasLeft = (left != NULL);
  bool hasRight = (right != NULL);
  bool hasParent = (parent != NULL);

  ar(CEREAL_NVP(hasLeft));
  ar(CEREAL_NVP(hasRight));
  ar(CEREAL_NVP(hasParent));
  if (hasLeft)
    ar(CEREAL_POINTER(left));
  if (hasRight)
    ar(CEREAL_POINTER(right));
  if (!hasParent)
  {
    MatType*& datasetTemp = const_cast<MatType*&>(dataset);
    ar(CEREAL_POINTER(datasetTemp));
  }

  if (cereal::is_loading<Archive>())
  {
    if (left)
      left->parent = this;
    if (right)
      right->parent = this;
  }
  // If we are the root, we need to restore the dataset pointer throughout
  if (!hasParent)
  {
    std::stack<BinarySpaceTree*> stack;
    if (left)
      stack.push(left);
    if (right)
      stack.push(right);
    while (!stack.empty())
    {
      BinarySpaceTree* node = stack.top();
      stack.pop();
      node->dataset = dataset;
      if (node->left)
        stack.push(node->left);
      if (node->right)
       stack.push(node->right);
    }
  }
}

} // namespace mlpack

#endif
