/**
 * @file rectangle_tree_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of generalized rectangle tree.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_RECTANGLE_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "rectangle_tree.hpp"

#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/string_util.hpp>

namespace mlpack {
namespace tree {

template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
RectangleTree<SplitType, DescentType, StatisticType, MatType>::RectangleTree(
        MatType& data,
        const size_t maxLeafSize,
        const size_t minLeafSize,
        const size_t maxNumChildren,
        const size_t minNumChildren,
        const size_t firstDataIndex) :
maxNumChildren(maxNumChildren),
minNumChildren(minNumChildren),
numChildren(0),
children(maxNumChildren + 1), // Add one to make splitting the node simpler
parent(NULL),
begin(0),
count(0),
maxLeafSize(maxLeafSize),
minLeafSize(minLeafSize),
bound(data.n_rows),
parentDistance(0),
dataset(data),
points(maxLeafSize + 1), // Add one to make splitting the node simpler.
localDataset(new MatType(data.n_rows, static_cast<int> (maxLeafSize) + 1)) // Add one to make splitting the node simpler
{
  stat = StatisticType(*this);

  // For now, just insert the points in order.
  RectangleTree* root = this;

  //for(int i = firstDataIndex; i < 57; i++) { // 56,57 are the bound for where it works/breaks
  for (size_t i = firstDataIndex; i < data.n_cols; i++) {
    root->InsertPoint(i);
  }
}

template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
RectangleTree<SplitType, DescentType, StatisticType, MatType>::RectangleTree(
        RectangleTree<SplitType, DescentType, StatisticType, MatType>* parentNode) :
maxNumChildren(parentNode->MaxNumChildren()),
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
dataset(parentNode->Dataset()),
points(maxLeafSize + 1), // Add one to make splitting the node simpler.
localDataset(new MatType(static_cast<int> (parentNode->Bound().Dim()), static_cast<int> (maxLeafSize) + 1)) // Add one to make splitting the node simpler
{
  stat = StatisticType(*this);
}

/**
 * Deletes this node, deallocating the memory for the children and calling
 * their destructors in turn.  This will invalidate any pointers or references
 * to any nodes which are children of this one.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
RectangleTree<SplitType, DescentType, StatisticType, MatType>::
~RectangleTree()
{
  for (int i = 0; i < numChildren; i++) {
    delete children[i];
  }
  //if(numChildren == 0)
  delete localDataset;
}

/**
 * Deletes this node but leaves the children untouched.  Needed for when we
 * split nodes and remove nodes (inserting and deleting points).
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::
SoftDelete()
{
  //if(numChildren != 0)
  //dataset = NULL;
  parent = NULL;
  for (int i = 0; i < children.size(); i++) {
    children[i] = NULL;
  }
  numChildren = 0;
  delete this;
}

/**
 * Set the dataset to null.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::
NullifyData()
{
  localDataset = NULL;
}

/**
 * Recurse through the tree and insert the point at the leaf node chosen
 * by the heuristic.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::
InsertPoint(const size_t point)
{
  // Expand the bound regardless of whether it is a leaf node.
  bound |= dataset.col(point);

  // If this is a leaf node, we stop here and add the point.
  if (numChildren == 0) {
    localDataset->col(count) = dataset.col(point);
    points[count++] = point;
    SplitNode();
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  children[DescentType::ChooseDescentNode(this, dataset.col(point))]->InsertPoint(point);
}

/**
 * Recurse through the tree to remove the point.  Once we find the point, we
 * shrink the rectangles if necessary.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
bool RectangleTree<SplitType, DescentType, StatisticType, MatType>::
DeletePoint(const size_t point)
{
  if (numChildren == 0) {
    for (size_t i = 0; i < count; i++) {
      if (points[i] == point) {
        localDataset->col(i) = localDataset->col(--count); // decrement count
        points[i] = points[count];
        CondenseTree(dataset.col(point)); // This function will ensure that minFill is satisfied.
        return true;
      }
    }
  }
  for (size_t i = 0; i < numChildren; i++) {
    if (children[i]->Bound().Contains(dataset.col(point)))
      if (children[i]->DeletePoint(point))
        return true;
  }
  return false;
}

template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
TreeSize() const
{
  int n = 0;
  for (int i = 0; i < numChildren; i++) {
    n += children[i]->TreeSize();
  }
  return n + 1; // we add one for this node
}

template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
TreeDepth() const
{
  /* Because R trees are balanced, we could simplify this.  However, X trees are not
     guaranteed to be balanced so I keep it as is: */
  int n = 1;
  RectangleTree<SplitType, DescentType, StatisticType, MatType>* currentNode =
          const_cast<RectangleTree*> (this);
  while (!currentNode->IsLeaf()) {
    currentNode = currentNode->Children()[0];
    n++;
  }
  return n;
}

template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline bool RectangleTree<SplitType, DescentType, StatisticType, MatType>::
IsLeaf() const
{
  return numChildren == 0;
}

/**
 * Return a bound on the furthest point in the node form the centroid.
 * This returns 0 unless the node is a leaf.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline double RectangleTree<SplitType, DescentType, StatisticType, MatType>::
FurthestPointDistance() const
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
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline double RectangleTree<SplitType, DescentType, StatisticType, MatType>::
FurthestDescendantDistance() const
{
  //return the distance from the centroid to a corner of the bound.
  return 0.5 * bound.Diameter();
}

/**
 * Return the number of points contained in this node.  Zero if it is a non-leaf node.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
NumPoints() const
{
  if (numChildren != 0) // This is not a leaf node.
    return 0;

  return count;
}

/**
 * Return the number of descendants under or in this node.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
NumDescendants() const
{
  if (numChildren == 0)
    return count;
  else {
    size_t n = 0;
    for (int i = 0; i < numChildren; i++) {
      n += children[i]->NumDescendants();
    }
    return n;
  }
}

/**
 * Return the index of a particular descendant contained in this node.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
Descendant(const size_t index) const
{
  return (points[index]);
}

/**
 * Return the index of a particular point contained in this node.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
Point(const size_t index) const
{
  return dataset(points[index]);
}

/**
 * Return the last point in the tree.  WARNING: POINTS ARE NOT MOVED IN THE ORIGINAL DATASET,
 * SO THIS IS RATHER POINTLESS.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::End() const
{
  if (numChildren)
    return begin + count;
  return children[numChildren - 1]->End();
}

//have functions for returning the list of modified indices if we end up doing it that way.

/**
 * Split the tree.  This calls the SplitType code to split a node.  This method should only
 * be called on a leaf node.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::SplitNode()
{
  // This should always be a leaf node.  When we need to split other nodes,
  // the split will be called from here but will take place in the SplitType code.
  assert(numChildren == 0);

  // Check to see if we are full.
  if (count < maxLeafSize)
    return; // We don't need to split.

  // If we are full, then we need to split (or at least try).  The SplitType takes
  // care of this and of moving up the tree if necessary.
  SplitType::SplitLeafNode(this);
}

/**
 * Condense the tree.  This shrinks the bounds and moves up the tree if applicable.
 * If a node goes below minimum fill, this code will deal with it.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::CondenseTree(const arma::vec& point)
{
  
  // First delete the node if we need to.  There's no point in shrinking the bound first.
  if (IsLeaf() && count < minLeafSize && parent != NULL) { //We can't delete the root node
    for (size_t i = 0; i < parent->NumChildren(); i++) {
      if (parent->Children()[i] == this) {
        parent->Children()[i] = parent->Children()[--parent->NumChildren()]; // decrement numChildren
        parent->ShrinkBoundForBound(bound); // We want to do this before reinserting points.

        // Reinsert the points at the root node.
        RectangleTree<SplitType, DescentType, StatisticType, MatType>* root = parent;
        while (root->Parent() != NULL)
          root = root->Parent();
        
        for (size_t j = 0; j < count; j++) {
          root->InsertPoint(points[j]);
        }

        parent->CondenseTree(point); // This will check the MinFill of the parent.
        //Now it should be safe to delete this node.
        SoftDelete();
        return;
      }
    }
    // Control should never reach here.
    assert(true == false);
  } else if (!IsLeaf() && numChildren < minNumChildren) {

    if (parent != NULL) { // The normal case.  We need to be careful with the root.
      for (size_t j = 0; j < parent->NumChildren(); j++) {
        if (parent->Children()[j] == this) {
          parent->Children()[j] = parent->Children()[--parent->NumChildren()]; // decrement numChildren
          parent->ShrinkBoundForBound(bound); // We want to do this before reinserting nodes.          

          size_t level = TreeDepth();
          // Reinsert the nodes at the root node.
          RectangleTree<SplitType, DescentType, StatisticType, MatType>* root = this;
          while (root->Parent() != NULL)
            root = root->Parent();
          for (size_t i = 0; i < numChildren; i++)
            root->InsertNode(children[i], level);

          parent->CondenseTree(point); // This will check the MinFill of the parent.
          //Now it should be safe to delete this node.
          SoftDelete();
          return;
        }
      }
    } else if (numChildren == 1) { // If there are multiple children, we can't do anything to the root.
      RectangleTree<SplitType, DescentType, StatisticType, MatType>* child = children[0];
      for (size_t i = 0; i < child->NumChildren(); i++) {
        children[i] = child->Children()[i];
      }
      numChildren = child->NumChildren();
      for (size_t i = 0; i < child->Count(); i++) { // In case the tree has a height of two.
        points[i] = child->Points()[i];
        localDataset->col(i) = child->LocalDataset().col(i);
      }
      count = child->Count();
      child->SoftDelete();
    }
  }

  // If we didn't delete it, shrink the bound if we need to.
  if (ShrinkBoundForPoint(point) && parent != NULL) {
    parent->CondenseTree(point);
  }

}

/**
 * Shrink the bound so it fits tightly after the removal of this point.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
bool RectangleTree<SplitType, DescentType, StatisticType, MatType>::ShrinkBoundForPoint(const arma::vec& point)
{
  bool shrunk = false;
  if (IsLeaf()) {
    for (size_t i = 0; i < bound.Dim(); i++) {
      if (bound[i].Lo() == point[i]) {
        double min = DBL_MAX;
        for (size_t j = 0; j < count; j++) {
          if (localDataset->col(j)[i] < min)
            min = localDataset->col(j)[i];
        }
        if (bound[i].Lo() < min) {
          shrunk = true;
          bound[i].Lo() = min;
        } else if(min < bound[i].Lo()) {
          assert(true == false); // we have a problem.
        }
      } else if (bound[i].Hi() == point[i]) {
        double max = -1 * DBL_MAX;
        for (size_t j = 0; j < count; j++) {
          if (localDataset->col(j)[i] > max)
            max = localDataset->col(j)[i];
        }
        if (bound[i].Hi() > max) {
          shrunk = true;
          bound[i].Hi() = max;
        } else if(max > bound[i].Hi()) {
          assert(true == false); // we have a problem.
        }
      }
    }
  } else {
    for (size_t i = 0; i < point.n_elem; i++) {
      if (bound[i].Lo() == point[i]) {
        double min = DBL_MAX;
        double max = -1 * DBL_MAX;
        for (size_t j = 0; j < numChildren; j++) {
          if (children[j]->Bound()[i].Lo() < min)
            min = children[j]->Bound()[i].Lo();
          if (children[j]->Bound()[i].Hi() > max)
            max = children[j]->Bound()[i].Hi();
        }
        if (bound[i].Lo() < min) {
          shrunk = true;
          bound[i].Lo() = min;
        }
        if (bound[i].Hi() > max) {
          shrunk = true;
          bound[i].Hi() = max;
        }
      }
    }
  }
  return shrunk;
}

/**
 * Shrink the bound so it fits tightly after the removal of this bound.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
bool RectangleTree<SplitType, DescentType, StatisticType, MatType>::ShrinkBoundForBound(const HRectBound<>& /* unused */)
{
  double sum = 0; // Using the sum is safe since none of the dimensions can increase.
  //I think it may be faster to just recalculate the whole thing.
  for (size_t i = 0; i < bound.Dim(); i++) {
    sum += bound[i].Width();
    bound[i].Lo() = DBL_MAX;
    bound[i].Hi() = -1 * DBL_MAX;
  }
  for (size_t i = 0; i < numChildren; i++) {
    bound |= children[i]->Bound();
  }
  double sum2 = 0;
  for (size_t i = 0; i < bound.Dim(); i++)
    sum2 += bound[i].Width();
  
  return sum != sum2;
}

/**
 * Insert the node into the tree at the appropriate depth.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
void RectangleTree<SplitType, DescentType, StatisticType, MatType>::InsertNode(
        const RectangleTree<SplitType, DescentType, StatisticType, MatType>* node,
        const size_t level)
{
  // Expand the bound regardless of the level.
  bound |= node->Bound();

  if (level == TreeDepth()) {
    children[numChildren++] = const_cast<RectangleTree*> (node);
    assert(numChildren <= maxNumChildren); // We should never have increased without splitting.
    if (numChildren == maxNumChildren)
      SplitType::SplitNonLeafNode(this);
  } else {
    children[DescentType::ChooseDescentNode(this, node)]->InsertNode(node, level);
  }
}

/**
 * Returns a string representation of this object.
 */
template<typename SplitType,
typename DescentType,
typename StatisticType,
typename MatType>
std::string RectangleTree<SplitType, DescentType, StatisticType, MatType>::ToString() const
{
  std::ostringstream convert;
  convert << "RectangleTree [" << this << "]" << std::endl;
//  convert << "  First point: " << begin << std::endl;
//  convert << "  Number of descendants: " << numChildren << std::endl;
//  convert << "  Number of points: " << count << std::endl;
//  convert << "  Bound: " << std::endl;
//  convert << mlpack::util::Indent(bound.ToString(), 2);
//  convert << "  Statistic: " << std::endl;
//  //convert << mlpack::util::Indent(stat.ToString(), 2);
//  convert << "  Max leaf size: " << maxLeafSize << std::endl;
//  convert << "  Min leaf size: " << minLeafSize << std::endl;
//  convert << "  Max num of children: " << maxNumChildren << std::endl;
//  convert << "  Min num of children: " << minNumChildren << std::endl;
//  convert << "  Parent address: " << parent << std::endl;
//
//  // How many levels should we print?  This will print 3 levels (counting the root).
//  if (parent == NULL || parent->Parent() == NULL) {
//    for (int i = 0; i < numChildren; i++) {
//      convert << children[i]->ToString();
//    }
//  }
  return convert.str();
}

}; //namespace tree
}; //namespace mlpack

#endif
