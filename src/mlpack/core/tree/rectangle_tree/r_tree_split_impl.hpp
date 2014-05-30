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
void RTreeSplit<MatType>::SplitLeafNode(const RectangleTree& tree)
{
  // Use the quadratic split method from: Guttman "R-Trees: A Dynamic Index Structure for
  // Spatial Searching"  It is simplified since we don't handle rectangles, only points.
  // It assumes that the tree uses Euclidean Distance.
  int i = 0;
  int j = 0;
  GetPointSeeds(tree, &i, &j);

  RectangleTree treeOne = new RectangleTree();
  treeOne.insertPoint(tree.dataset[i]);
  RectangleTree treeTwo = new RectangleTree();
  treeTwo.insertPoint(tree.dataset[j]);

  AssignPointDestNode(tree, treeOne, treeTwo, i, j);
  
  RectangleTree* par = tree.parent();
  int index = 0;
  for(int i = 0; i < par.numOfChildren(); i++) {
    if(par.getChildren()[i] == this) {
      index = i;
      break;
    }
  }
  par.getChildren()[i] = treeOne;
  par.getChildren()[par.end++] = treeTwo;

  delete tree;

  // we only add one at a time, so should only need to test for equality
  // just in case, we use an assert.
  boost::assert(numChildren <= maxNumChildren);

  if(par.numOfChildren() == par.maxNumChildren) {
    SplitNonLeafNode(par);
  }
  return;
}

bool RTreeSplit<MatType>::SplitNonLeafNode(const RectangleTree& tree)
{
  int i = 0;
  int j = 0;
  GetBoundSeeds(tree, &i, &j);
  
  RectangleTree treeOne = new RectangleTree();
  treeOne.insertRectangle()
  RectangleTree treeTwo = new RectangleTree();
  
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
    for(int j = i+1; j < tree.count; j++) {
      double score = 1.0;
      for(int k = 0; k < dimensions; k++) {
	score *= std::abs(tree.dataset[i][k] - tree.dataset[j][k]);
      }
      if(score > worstPairScore) {
	worstPairScore = score;
	worstI = i;
	worstJ = j;
      }
    }
  }

  *iRet = worstI;
  *jRet = worstJ;
  return;
}

/**
 * Get the two bounds that will be used as seeds for the split of the node.
 * The indices of the bounds will be stored in iRet and jRet.
 */
void RTreeSplit<MatType>::GetBoundSeeds(const RectangleTree& tree, int* iRet, int* jRet)
{
  double worstPairScore = 0.0;
  int worstI = 0;
  int worstJ = 0;
  for(int i = 0; i < tree.numChildren; i++) {
    for(int j = i+1; j < tree.numChildren; j++) {
      double score = 1.0;
      for(int k = 0; k < dimensions; k++) {
	score *= std::max(tree.children[i].bound[k].hi(), tree.children[j].bound[k].hi) - 
	  std::min(tree.children[i].bound[k].low(), tree.children[j].bound[k].low());
      }
      if(score > worstPairScore) {
	worstPairScore = score;
	worstI = i;
	worstJ = j;
      }
    }
  }

  *iRet = worstI;
  *jRet = worstJ;
  return;
}


void RTreeSplit<MatType>::AssignPointDestNode(
    const RectangleTree& oldTree,
    RectangleTree& treeOne,
    RectangleTree& treeTwo,
    const int intI,
    const int intJ)
{
  Log::assert(end > 1); // If this isn't true, the tree is really weird.
  int end = oldTree.count;
  oldTree.dataset.col(intI) = oldTree.dataset.col(--end); // decrement end
  oldTree.dataset.col(intJ) = oldTree.dataset.col(--end); // decrement end
    
  int index = 0;

  // In each iteration, we go through all points and find the one that causes the least
  // increase of volume when added to one of the rectangles.  We then add it to that
  // rectangle.  
  while() {
    int bestIndex = 0;
    double bestScore = 0;
    int bestRect = 0;

    // Calculate the increase in volume for assigning this point to each rectangle.
    double volOne = 1.0;
    double volTwo = 1.0;
    for(int i = 0; i < bound.Dim(); i++) {
      volOne *= treeOne.bound[i].width();
      volTwo *= treeTwo.bound[i].width();
    }

    for(int j = 0; j < end; j++) {
      double newVolOne = 1.0;
      double newVolTwo = 1.0;
      for(int i = 0; i < bound.Dim(); i++) {
	double c = oldTree.dataset.col(index)[i];      
	newVolOne *= treeOne.bound[i].contains(c) ? treeOne.bound[i].width() :
	  (c < treeOne.bound[i].low() ? (high - c) : (c - low));
	newVolTwo *= treeTwo.bound[i].contains(c) ? treeTwo.bound[i].width() :
	  (c < treeTwo.bound[i].low() ? (high - c) : (c - low));
      }
    
      if((newVolOne - volOne) < (newVolTwo - volTwo)) {
	if(newVolOne - volOne < bestScore) {
	  bestScore = newVolOne - volOne;
	  bestIndex = index;
	  bestRect = 1;
	}
      } else {
	if(newVolTwo - volTwo < bestScore) {
	  bestScore = newVolTwo - volTwo;
	  bestIndex = index;
	  bestRect = 2;
	}
      }
    }

    // Assign the point that causes the least increase in volume 
    // to the appropriate rectangle.
    if(bestRect == 1)
      treeOne.insertPoint(oldTree.dataset(bestIndex);
    else
      treeTwo.insertPoint(oldTree.dataset(bestIndex);

    oldTree.dataset.col(bestIndex) = oldTree.dataset.col(--end); // decrement end.
  }


}





}; // namespace tree
}; // namespace mlpack

#endif
