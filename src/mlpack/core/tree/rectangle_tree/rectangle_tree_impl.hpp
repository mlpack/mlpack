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

template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
RectangleTree<StatisticType, MatType, SplitType, DescentType>::RectangleTree(
    MatType& data,
    const size_t maxLeafSize,
    const size_t minLeafSize,
    const size_t maxNumChildren,
    const size_t minNumChildren,
    const size_t firstDataIndex = 0):

{
  this.maxNumChildren = maxNumChildren;
  this.minNumChildren = minNumChildren;
  this.numChildren = 0;
  this.parent = NULL;
  this.begin = 0;
  this.count = 0;
  this.maxLeafSize = maxLeafSize;
  this.minLeafSize = minLeafSize;
  this.bound = new HRectBound(data.n_rows);
  this.stat = EmptyStatistic;
  this.parentDistance = 0.0;
  this.furthestDescendantDistance = 0.0;
  this.dataset = new MatType(maxLeafSize+1); // Add one to make splitting the node simpler
  this.children = new std::vector<RectangleTree*>(maxNumChildren+1); // ibid.
  
  // For now, just insert the points in order.
  RectangleTree* root = this;
  for(int i = firstDataIndex; i < n_cols; i++) {
    root.insertPoint(data.col(i));
    if(root.Parent() != NULL) {
      root = root.Parent(); // OK since the level increases by at most one per iteration.
    }
  }
  
}

/**
 * Deletes this node, deallocating the memory for the children and calling
 * their destructors in turn.  This will invalidate any pointers or references
 * to any nodes which are children of this one.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
RectangleTree<BoundType, StatisticType, MatType, SplitType, DescentType>::
  ~RectangleTree()
{
  for(int i = 0; i < numOfChildren; i++) {
    delete children[i];
  }
}


/**
 * Recurse through the tree and insert the point at the leaf node chosen
 * by the heuristic.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    InsertPoint(const arma::vec& point)
{
  // Expand the bound regardless of whether it is a leaf node.
  bound |= point;
  
  // If this is a leaf node, we stop here and add the point.
  if(numChildren == 0) {
    data.col(points++) = point;
    splitNode();
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  double minScore = DescentType.EvalNode(children[0].bound, point);
  int bestIndex = 0;
  for(int i = 1; i < numChildren; i++) {
    double score = DescentType.EvalNode(children[i].bound, point);
    if(score < minScore) {
      minScore = score;
      bestIndex = i
    }
  }
  children[bestIndex].InsertPoint(point);
}

template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    TreeSize() const
{
  int n = 0;
  for(int i = 0; i < numOfChildren; i++) {
    n += children[i].TreeSize();
  }
  return n + 1; // we add one for this node
}



template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    TreeDepth() const
{
  // Recursively count the depth of each subtree.  The plus one is
  // because we have to count this node, too.
  int maxSubDepth = 0;
  for(int i = 0; i < numOfChildren; i++) {
    int d = children[i].depth();
    if(d > maxSubDepth)
      maxSubDepth = d;
  }
  return maxSubDepth + 1;
}

template<typename StatisticType,
         typename MatType,
         typename SplitType,
	typename DescentType>
inline bool BinarySpaceTree<StatisticType, MatType, SplitType, DescentType>::
    IsLeaf() const
{
  return numOfChildren == 0;
}

/**
 * Returns the number of children in this node.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    NumChildren() const
{
  return NumChildren;
}

/**
 * Return a bound on the furthest point in the node form the centroid.
 * This returns 0 unless the node is a leaf.
 */
template<typename StatisticType,
         typename MatType,
         typename SplitType,
	typename DescentType>
inline double RectangleTree<StatisticType, MatType, SplitType, DescentType>::
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
template<typename StatisiticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline double RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    FurthestDescendantDistance() const
{
  return furthestDescendantDistance;
}

/**
 * Return the specified child.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline RectangleTree<StatisticType, MatType, SplitType, DescentType>&
    RectangleTree<StatisticType, MatType, SplitType, DescentType>::
        Child(const size_t child) const
{
  return children[child];
}

/**
 * Return the number of points contained in this node.  Zero if it is not a leaf.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    NumPoints() const
{
  if(numChildren == 0)
    return 0;

  return count;
}

/**
 * Return the number of descendants contained in this node.  MEANINIGLESS AS IT CURRENTLY STANDS.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    NumDescendants() const
{
  return count;
}

/**
 * Return the index of a particular descendant contained in this node.  SEE OTHER WARNINGS
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    Descendant(const size_t index> const
{
  return (begin + index);
}

/**
 * Return the index of a particular point contained in this node.  SEE OTHER WARNINGS
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::
    Point(const size_t index> const
{
  return (begin + index);
}

/**
 * Return the last point in the tree.  SINCE THE TREE STORES DATA SEPARATELY IN EACH LEAF
 * THIS IS CURRENTLY MEANINGLESS.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
inline size_t RectangleTree<StatisticType, MatType, SplitType, DescentType>::End() const
{
  if(numOfChildren)
    return begin + count;
  return children[numOfChildren-1].End();
}

  //have functions for returning the list of modified indices if we end up doing it that way.

/**
 * Split the tree.  This calls the SplitType code to split a node.  This method should only
 * be called on a leaf node.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType,
	 typename DescentType>
void RectangleTree<StatisticType, MatType, SplitType, DescentType>::SplitNode(
    RetangleTree& tree)
{
  // This should always be a leaf node.  When we need to split other nodes,
  // the split will be called from here but will take place in the SplitType code.
  boost::assert(numChildren == 0);

  // See if we are full.
  if(points < maxLeafSize)
    return;
  
  // If we are full, then we need to move up the tree.  The SplitType takes
  // care of this.
  SplitType.SplitLeafNode(this);    
}


/**
 * Returns a string representation of this object.
 */
  template<typename StatisticType,
	   typename MatType,
	   typename SplitType,
	 typename DescentType>
std::string BinarySpaceTree<StatisticType, MatType, SplitType, DescentType>::ToString() const
{
  std::ostringstream convert;
  convert << "RectangleTree [" << this << "]" << std::endl;
  convert << "  First point: " << begin << std::endl;
  convert << "  Number of descendants: " << count << std::endl;
  convert << "  Bound: " << std::endl;
  convert << mlpack::util::Indent(bound.ToString(), 2);
  convert << "  Statistic: " << std::endl;
  convert << mlpack::util::Indent(stat.ToString(), 2);
  convert << "  Max leaf size: " << maxLeafSize << std::endl;
  convert << "  Split dimension: " << splitDimension << std::endl;

  // How many levels should we print?  This will print the root and it's children.
  if(parent == NULL) {
    for(int i = 0; i < numChildren; i++) {
      children[i].ToString();
    }
  }
}

}; //namespace tree
}; //namespace mlpack

#endif
