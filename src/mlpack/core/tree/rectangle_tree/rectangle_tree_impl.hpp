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
    const size_t firstDataIndex):
    maxNumChildren(maxNumChildren),
    minNumChildren(minNumChildren),
    numChildren(0),
    children(maxNumChildren+1), // Add one to make splitting the node simpler
    parent(NULL),
    begin(0),
    count(0),
    maxLeafSize(maxLeafSize),
    minLeafSize(minLeafSize),
    bound(data.n_rows),
    parentDistance(0),
    dataset(data),
    points(maxLeafSize+1) // Add one to make splitting the node simpler.
{
  stat = StatisticType(*this);

  // For now, just insert the points in order.
  RectangleTree* root = this;

  //for(int i = firstDataIndex; i < 57; i++) { // 56,57 are the bound for where it works/breaks
  for(size_t i = firstDataIndex; i < data.n_cols; i++) {
    root->InsertPoint(i);
  }
}

template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
RectangleTree<SplitType, DescentType, StatisticType, MatType>::RectangleTree(
  RectangleTree<SplitType, DescentType, StatisticType, MatType>* parentNode):
  maxNumChildren(parentNode->MaxNumChildren()),
  minNumChildren(parentNode->MinNumChildren()),
  numChildren(0),
  children(maxNumChildren+1),
  parent(parentNode),
  begin(0),
  count(0),
  maxLeafSize(parentNode->MaxLeafSize()),
  minLeafSize(parentNode->MinLeafSize()),
  bound(parentNode->Bound().Dim()),
  parentDistance(0),
  dataset(parentNode->Dataset()),
  points(maxLeafSize+1) // Add one to make splitting the node simpler.
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
  for(int i = 0; i < numChildren; i++) {
    delete children[i];
  }
  //if(numChildren == 0)
  //delete points;
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
    softDelete()
{
  //if(numChildren != 0)
    //dataset = NULL;
  parent = NULL;
  for(int i = 0; i < children.size(); i++) {
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
  //points = NULL;
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
  if(numChildren == 0) {
    points[count++] = point;
    SplitNode();
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  double minScore = DescentType::EvalNode(children[0]->Bound(), dataset.col(point));
  int bestIndex = 0;

  for(int i = 1; i < numChildren; i++) {
    double score = DescentType::EvalNode(children[i]->Bound(), dataset.col(point));
    if(score < minScore) {
      minScore = score;
      bestIndex = i;
    }
  }
  children[bestIndex]->InsertPoint(point);
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
  if(numChildren == 0) {
    for(int i = 0; i < count; i++) {
      if(points[i] == point) {
	points[i] = points[--count];
	for(int j = 0; j < bound.Dim(); j++) {
	  if(bound[j].Lo() == dataset.col(point)[j]) {
	    int loIndx = 0;
	    double lo = dataset(points[0])[j];
	    for(int k = 1; k < count; k++) {
	      if(dataset(points[k])[j] < lo) {
		lo = dataset(points[k])[j];
		loIndx = k;
	      }
	    }
	    bound[j].Lo() = lo;
	  } else if(bound[j].Hi() == dataset.col(point)[j]) {
	    int hiIndx = 0;
	    double hi = dataset(points[0])[j];
	    for(int k = 1; k < count; k++) {
	      if(dataset(points[k])[j] > hi) {
		hi = dataset(points[k])[j];
		hiIndx = k;
	      }
	    }
	    bound[j].Hi() = hi;
	  }
	}
	return true;
      }
    }
  } else {
      for(int i = 0; i < numChildren; i++) {
	if(children[i].Bound().Contains(dataset.col(point))) {
	  if(children[i].DeletePoint(dataset.col(point))) {

	    return true;
	  }
	}
      }
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
  for(int i = 0; i < numChildren; i++) {
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

  // Recursively count the depth of each subtree.  The plus one is
  // because we have to count this node, too.
  int maxSubDepth = 0;
  for(int i = 0; i < numChildren; i++) {
    int d = children[i]->TreeDepth();
    if(d > maxSubDepth)
      maxSubDepth = d;
  }
  return maxSubDepth + 1;
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
  if(!IsLeaf())
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
  if(numChildren != 0) // This is not a leaf node.
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
  if(numChildren == 0)
    return count;
  else {
    size_t n = 0;
    for(int i = 0; i < numChildren; i++) {
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
  if(numChildren)
    return begin + count;
  return children[numChildren-1]->End();
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
  if(count < maxLeafSize)
    return; // We don't need to split.

  // If we are full, then we need to split (or at least try).  The SplitType takes
  // care of this and of moving up the tree if necessary.
  SplitType::SplitLeafNode(this);
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
  convert << "  First point: " << begin << std::endl;
  convert << "  Number of descendants: " << numChildren << std::endl;
  convert << "  Number of points: " << count << std::endl;
  convert << "  Bound: " << std::endl;
  convert << mlpack::util::Indent(bound.ToString(), 2);
  convert << "  Statistic: " << std::endl;
  //convert << mlpack::util::Indent(stat.ToString(), 2);
  convert << "  Max leaf size: " << maxLeafSize << std::endl;
  convert << "  Min leaf size: " << minLeafSize << std::endl;
  convert << "  Max num of children: " << maxNumChildren << std::endl;
  convert << "  Min num of children: " << minNumChildren << std::endl;
  convert << "  Parent address: " << parent << std::endl;

  // How many levels should we print?  This will print 3 levels (counting the root).
  if(parent == NULL || parent->Parent() == NULL) {
    for(int i = 0; i < numChildren; i++) {
      convert << children[i]->ToString();
    }
  }
  return convert.str();
}

}; //namespace tree
}; //namespace mlpack

#endif
