/**
 * @file rectangle_tree_impl.hpp
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
	 typename SplitType
	 typename DescentType>
RectangleTree<StatisticType, MatType, SplitType>::RectangleTree(
    MatType& data,
    const size_t leafSize):

{
  //Do the actual stuff here

}

/**
 * Recurse through the tree and insert the point at the leaf node chosen
 * by the heuristic.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType
	 typename HueristicType>
RectangleTree<StatisticType, MatType, SplitType, HueristicType>::
    InsertPoint(const arma::vec& point)
{
  if(numChildren == 0) {
    data.col(points++) = point;
    return;
  }
  double minScore = HueristicType.EvalNode(children[0].bound, point);
  int bestIndex = 0;
  for(int i = 1; i < numChildren; i++) {
    double score = HueristicType.EvalNode(children[i].bound, point);
    if(score < minScore) {
      minScore = score;
      bestIndex = i
    }
  }
  children[bestIndex].InsertPoint(point);
}


/**
 * Deletes this node, deallocating the memory for the children and calling
 * their destructors in turn.  This will invalidate any pointers or references
 * to any nodes which are children of this one.
 */
template<typename StatisticType,
	   typename MatType,
	   typename SplitType>
RectangleTree<BoundType, StatisticType, MatType, SplitType>::
  ~RectangleTree()
{
  for(int i = 0; i < numOfChildren; i++) {
    delete children[i];
  }
}

template<typename StatisticType,
	 typename MatType,
	 typename SplitType>
size_t RectangleTree<StatisticType, MatType, SplitType>::
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
	 typename SplitType>
size_t RectangleTree<StatisticType, MatType, SplitType>::
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
         typename SplitType>
inline bool BinarySpaceTree<StatisticType, MatType, SplitType>::
    IsLeaf() const
{
  return numOfChildren == 0;
}

/**
 * Returns the number of children in this node.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType>
inline size_t RectangleTree<StatisticType, MatType, SplitType>::
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
         typename SplitType>
inline double RectangleTree<StatisticType, MatType, SplitType>::
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
	 typename SplitType>
inline double RectangleTree<StatisticType, MatType, SplitType>::
    FurthestDescendantDistance() const
{
  return furthestDescendantDistance;
}

/**
 * Return the specified child.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType>
inline RectangleTree<StatisticType, MatType, SplitType>&
    RectangleTree<StatisticType, MatType, SplitType>::
        Child(const size_t child) const
{
  return children[child];
}

/**
 * Return the number of points contained in this node.  Zero if it is not a leaf.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType>
inline size_t RectangleTree<StatisticType, MatType, SplitType>::
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
	 typename SplitType>
inline size_t RectangleTree<StatisticType, MatType, SplitType>::
    NumDescendants() const
{
  return count;
}

/**
 * Return the index of a particular descendant contained in this node.  SEE OTHER WARNINGS
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType>
inline size_t RectangleTree<StatisticType, MatType, SplitType>::
    Descendant(const size_t index> const
{
  return (begin + index);
}

/**
 * Return the index of a particular point contained in this node.  SEE OTHER WARNINGS
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType>
inline size_t RectangleTree<StatisticType, MatType, SplitType>::
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
	 typename SplitType>
inline size_t RectangleTree<StatisticType, MatType, SplitType>::End() const
{
  if(numOfChildren)
    return begin + count;
  return children[numOfChildren-1].End();
}

  //have functions for returning the list of modified indices if we end up doing it that way.

/**
 * Split the tree.  This moves up the tree recursively.
 */
template<typename StatisticType,
	 typename MatType,
	 typename SplitType>
void RectangleTree<StatisticType, MatType, SplitType>::SplitNode(
    RetangleTree& tree)
{
  
}


/**
 * Returns a string representation of this object.
 */
  template<typename StatisticType,
	   typename MatType,
	   typename SplitType>
std::string BinarySpaceTree<StatisticType, MatType, SplitType>::ToString() const
{
  std::ostringstream convert;
  convert << "RectangleTree [" << this << "]" << std::endl;
  convert << "  First point: " << begin << std::endl;
  convert << "  Number of descendants: " << count << std::endl;
  convert << "  Bound: " << std::endl;
  convert << mlpack::util::Indent(bound.ToString(), 2);
  convert << "  Statistic: " << std::endl;
  convert << mlpack::util::Indent(stat.ToString(), 2);
  convert << "  Leaf size: " << leafSize << std::endl;
  convert << "  Split dimension: " << splitDimension << std::endl;

  // How many levels should we print?  This will print the top two tree levels.
  for(
}

}; //namespace tree
}; //namespace mlpack

#endif
