/**
 * @file r_tree_split_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of class (RTreeSplit) to split a RectangleTree.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP

#include "r_tree_split.hpp"

namespace mlpack {
namespace tree {

template<typename MatType>
bool RTreeSplit<MatType>::SplitLeafNode(const RectangleTree& tree)
{
  // Use the quadratic split method from: Guttman "R-Trees: A Dynamic Index Structure for
  // Spatial Searching"  It is simplified since we don't handle rectangles, only points.
  // It assumes that the tree uses Euclidean Distance.
  int i = 0;
  int j = 0;
  GetPointSeeds(tree, &i, &j);

  RectangleTree treeOne = new RectangleTree();
  treeOne.insertPoint(tree.points[i]);
  RectangleTree treeTwo = new RectangleTree();
  treeTwo.insertPoint(tree.points[j]);

  int treeOneCount = 1;
  int treeTwoCount = 1;

  
  
}

bool RTreeSplit<MatType>::SplitNonLeafNode(const RectangleTree& tree)
{

}

/**
 * Get the two points that will be used as seeds for the split of a leaf node.
 * The indices of these points will be stored in iRet and jRet.
 */
void RTreeSplit<MatType>::GetPointSeeds(const RectangleTree& tree, int* iRet, int* jRet)
{
  // Here we want to find the pair of points that it is worst to place in the same
  // node.  Because we are just using points, we will simply choose the two that would
  // create the most voluminous hyperrectangle.
  double worstPairScore = 0.0;
  int worstI = 0;
  int worstJ = 0;
  for(int i = 0; i < tree.count; i++) {
    for(int j = i; j < tree.count; j++) {
      double score = 1.0;
      for(int k = 0; k < dimensions; k++) {
	score *= std::abs(tree.points[i][k] - tree.points[j][k]);
      }
      if(score > worstPairScore) {
	worstPairScore = score;
	worstI = i;
	worstJ = j;
      }
    }
  }

  *iRet = i;
  *jRet = j;
  return;
}



}; // namespace tree
}; // namespace mlpack

#endif
