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
    const size_t maxLeafSize = 20,
    const size_t minLeafSize = 6,
    const size_t maxNumChildren = 4,
    const size_t minNumChildren = 0,
    const size_t firstDataIndex = 0):
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
    parentDistance(0)

{
  this.stat = EmptyStatistic(*this);
  this.dataset = new MatType(maxLeafSize+1); // Add one to make splitting the node simpler
  
  // For now, just insert the points in order.
  RectangleTree* root = this;
  for(int i = firstDataIndex; i < data.n_cols; i++) {
    root.insertPoint(data.col(i));
    if(root.Parent() != NULL) {
      root = root.Parent(); // OK since the level increases by at most one per iteration.
    }
  }
  
}

template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
RectangleTree<SplitType, DescentType, StatisticType, MatType>::RectangleTree(
  const RectangleTree<SplitType, DescentType, StatisticType, MatType>& parentNode):
  maxNumChildren(parentNode.MaxNumChildren()),
  minNumChildren(parentNode.MinNumChildren()),
  numChildren(0),
  children(maxNumChildren+1),
  parent(&parentNode),
  begin(0),
  count(0),
  maxLeafSize(parentNode.MaxLeafSize()),
  minLeafSize(parentNode.MinLeafSize()),
  bound(parentNode.Bound().Dim()),
  parentDistance(0)
  {
    this.stat = EmptyStatistic(*this);
    this.dataset = new MatType(maxLeafSize+1); // Add one to make splitting the node simpler
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
  delete dataset;
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
    InsertPoint(const arma::vec& point)
{
  // Expand the bound regardless of whether it is a leaf node.
  bound |= point;
  
  // If this is a leaf node, we stop here and add the point.
  if(numChildren == 0) {
    dataset.col(count++) = point;
    SplitNode();
    return;
  }

  // If it is not a leaf node, we use the DescentHeuristic to choose a child
  // to which we recurse.
  double minScore = DescentType::EvalNode(children[0].bound, point);
  int bestIndex = 0;
  for(int i = 1; i < numChildren; i++) {
    double score = DescentType::EvalNode(children[i].bound, point);
    if(score < minScore) {
      minScore = score;
      bestIndex = i;
    }
  }
  children[bestIndex].InsertPoint(point);
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
    n += children[i].TreeSize();
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
  // Recursively count the depth of each subtree.  The plus one is
  // because we have to count this node, too.
  int maxSubDepth = 0;
  for(int i = 0; i < numChildren; i++) {
    int d = children[i].depth();
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
  return furthestDescendantDistance;
}

/**
 * Return the specified child.
 */
template<typename SplitType,
         typename DescentType,
	 typename StatisticType,
         typename MatType>
inline RectangleTree<SplitType, DescentType, StatisticType, MatType>&
    RectangleTree<SplitType, DescentType, StatisticType, MatType>::
        Child(const size_t child) const
{
  return children[child];
}

/**
 * Return the number of points contained in this node.  Zero if it is not a leaf.
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
    NumPoints() const
{
  if(numChildren == 0)
    return 0;

  return count;
}

/**
 * Return the number of descendants contained in this node.  MEANINIGLESS AS IT CURRENTLY STANDS.
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
    NumDescendants() const
{
  return count;
}

/**
 * Return the index of a particular descendant contained in this node.  SEE OTHER WARNINGS
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
    Descendant(const size_t index) const
{
  return (begin + index);
}

/**
 * Return the index of a particular point contained in this node.  SEE OTHER WARNINGS
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::
    Point(const size_t index) const
{
  return (begin + index);
}

/**
 * Return the last point in the tree.  SINCE THE TREE STORES DATA SEPARATELY IN EACH LEAF
 * THIS IS CURRENTLY MEANINGLESS.
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
inline size_t RectangleTree<SplitType, DescentType, StatisticType, MatType>::End() const
{
  if(numChildren)
    return begin + count;
  return children[numChildren-1].End();
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

  // See if we are full.
  if(count < maxLeafSize)
    return;
  
  // If we are full, then we need to move up the tree.  The SplitType takes
  // care of this.
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
  convert << mlpack::util::Indent(stat.ToString(), 2);
  convert << "  Max leaf size: " << maxLeafSize << std::endl;

  // How many levels should we print?  This will print the root and it's children.
  if(parent == NULL) {
    for(int i = 0; i < numChildren; i++) {
      convert << children[i].ToString();
    }
  }
  return convert.str();
}

}; //namespace tree
}; //namespace mlpack

#endif
