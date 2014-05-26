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
	 typename SplitType>
RectangleTree<StatisticType, MatType, SplitType>::RectangleTree(
    MatType& data,
    const size_t leafSize):

{
  //Do the actual stuff here

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
