/**
 * @file r_tree_split_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of class (RTreeSplit) to split a RectangleTree.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP

#include "r_tree_split.hpp"
#include <mlpack/core/math/range.hpp>

namespace mlpack {
namespace tree {

/**
 * We call GetPointSeeds to get the two points which will be the initial points in the new nodes
 * We then call AssignPointDestNode to assign the remaining points to the two new nodes.
 * Finally, we delete the old node and insert the new nodes into the tree, spliting the parent
 * if necessary.
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<SplitType, DescentType, StatisticType, MatType>::SplitLeafNode(
  const RectangleTree<SplitType, DescentType,  StatisticType, MatType>& tree)
{
  // Use the quadratic split method from: Guttman "R-Trees: A Dynamic Index Structure for
  // Spatial Searching"  It is simplified since we don't handle rectangles, only points.
  // It assumes that the tree uses Euclidean Distance.
  int i = 0;
  int j = 0;
  GetPointSeeds(tree, &i, &j);

  RectangleTree<SplitType, DescentType,  StatisticType, MatType> treeOne = new 
    RectangleTree<SplitType, DescentType,  StatisticType, MatType>(tree.Parent());
  RectangleTree<SplitType, DescentType,  StatisticType, MatType> treeTwo = new 
    RectangleTree<SplitType, DescentType,  StatisticType, MatType>(tree.Parent());
  
  // This will assign the ith and jth point appropriately.
  AssignPointDestNode(tree, treeOne, treeTwo, i, j);
  
  // create the parent node if necessary
  if(tree.Parent() == NULL) {
    
  }
  
  //Remove this node and insert treeOne and treeTwo
  RectangleTree<SplitType, DescentType,  StatisticType, MatType>* par = tree.Parent();
  int index = 0;
  for(int i = 0; i < par.numOfChildren(); i++) {
    if(par.getChildren()[i] == tree) {
      index = i;
      break;
    }
  }
  par.getChildren()[i] = treeOne;
  par.getChildren()[par.end++] = treeTwo;

  //because we copied the points to treeOne and treeTwo, we can just delete this node
  delete tree;

  // we only add one at a time, so should only need to test for equality
  // just in case, we use an assert.
  assert(par.NumChildren() <= par.MaxNumChildren());

  if(par.NumChildren() == par.MaxNumChildren()) {
    SplitNonLeafNode(par);
  }
  return;
}

/**
 * We call GetBoundSeeds to get the two new nodes that this one will be broken
 * into.  Then we call AssignNodeDestNode to move the children of this node
 * into either of those two nodes.  Finally, we delete the now unused information
 * and recurse up the tree if necessary.  We don't need to worry about the bounds
 * higher up the tree because they were already updated if necessary.
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
bool RTreeSplit<SplitType, DescentType, StatisticType, MatType>::SplitNonLeafNode(
  const RectangleTree<SplitType, DescentType,  StatisticType, MatType>& tree)
{
  int i = 0;
  int j = 0;
  GetBoundSeeds(tree, &i, &j);
  
  RectangleTree<SplitType, DescentType,  StatisticType, MatType> treeOne = new 
    RectangleTree<SplitType, DescentType,  StatisticType, MatType>(tree.Parent());
  RectangleTree<SplitType, DescentType,  StatisticType, MatType> treeTwo = new 
    RectangleTree<SplitType, DescentType,  StatisticType, MatType>(tree.Parent());
  
  // This will assign the ith and jth rectangles appropriately.
  AssignNodeDestNode(tree, treeOne, treeTwo, i, j);

  // create the parent node if necessary
  if(tree.Parent() == NULL) {
    tree.Parent() = new RectangleTree<SplitType, DescentType,  StatisticType, MatType>();
  }
  
  //Remove this node and insert treeOne and treeTwo
  RectangleTree<SplitType, DescentType,  StatisticType, MatType>* par = tree.parent();
  int index = 0;
  for(int i = 0; i < par.numOfChildren(); i++) {
    if(par.getChildren()[i] == tree) {
      index = i;
      break;
    }
  }
  par.getChildren()[i] = treeOne;
  par.getChildren()[par.end++] = treeTwo;

  // Because we now have pointers to the information stored under this tree,
  // we need to delete this node carefully.
  tree.softDelete();

  // we only add one at a time, so should only need to test for equality
  // just in case, we use an assert.
  assert(par.NumChildren() <= par.MaxNumChildren());

  if(par.NumChildren() == par.MaxNumChildren()) {
    SplitNonLeafNode(par);
  }
  return;
}

/**
 * Get the two points that will be used as seeds for the split of a leaf node.
 * The indices of these points will be stored in iRet and jRet.
 */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<SplitType, DescentType, StatisticType, MatType>::GetPointSeeds(
  const RectangleTree<SplitType, DescentType,  StatisticType, MatType>& tree,
  int* iRet,
  int* jRet)
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
      for(int k = 0; k < tree.Bound().Dim(); k++) {
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
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<SplitType, DescentType, StatisticType, MatType>::GetBoundSeeds(
  const RectangleTree<SplitType, DescentType,  StatisticType, MatType>& tree,
  int* iRet,
  int* jRet)
{
  double worstPairScore = 0.0;
  int worstI = 0;
  int worstJ = 0;
  for(int i = 0; i < tree.numChildren; i++) {
    for(int j = i+1; j < tree.numChildren; j++) {
      double score = 1.0;
      for(int k = 0; k < tree.Bound().Dim(); k++) {
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

template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<SplitType, DescentType, StatisticType, MatType>::AssignPointDestNode(
    const RectangleTree<SplitType, DescentType, StatisticType, MatType>& oldTree,
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& treeOne,
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& treeTwo,
    const int intI,
    const int intJ)
{
  int end = oldTree.count;
  assert(end > 1); // If this isn't true, the tree is really weird.

  treeOne.insertPoint(oldTree.dataset.col(intI));
  oldTree.dataset.col(intI) = oldTree.dataset.col(--end); // decrement end
  treeTwo.insertPoint(oldTree.dataset.col(intJ));
  oldTree.dataset.col(intJ) = oldTree.dataset.col(--end); // decrement end
  

  int numAssignedOne = 1;
  int numAssignedTwo = 1;

  // In each iteration, we go through all points and find the one that causes the least
  // increase of volume when added to one of the rectangles.  We then add it to that
  // rectangle.  We stop when we run out of points or when all of the remaining points
  // need to be assigned to the same rectangle to satisfy the minimum fill requirement.
  while(end > 1 && end < oldTree.minLeafSize() - std::min(numAssignedOne, numAssignedTwo)) {
    int bestIndex = 0;
    double bestScore = 0;
    int bestRect = 0;

    // Calculate the increase in volume for assigning this point to each rectangle.
    double volOne = 1.0;
    double volTwo = 1.0;
    for(int i = 0; i < oldTree.Bound().Dim(); i++) {
      volOne *= treeOne.bound[i].width();
      volTwo *= treeTwo.bound[i].width();
    }

    // Find the point that, when assigned to one of the two new rectangles, minimizes the increase
    // in volume.
    for(int index = 0; index < end; index++) {
      double newVolOne = 1.0;
      double newVolTwo = 1.0;
      for(int i = 0; i < oldTree.Bound().Dim(); i++) {
	double c = oldTree.dataset.col(index)[i];      
	newVolOne *= treeOne.bound[i].contains(c) ? treeOne.bound[i].width() :
	  (c < treeOne.bound[i].low() ? (treeOne.bound[i].high() - c) : (c - treeOne.bound[i].low()));
	newVolTwo *= treeTwo.bound[i].contains(c) ? treeTwo.bound[i].width() :
	  (c < treeTwo.bound[i].low() ? (treeTwo.bound[i].high() - c) : (c - treeTwo.bound[i].low()));
      }
    
      // Choose the rectangle that requires the lesser increase in volume.
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
      treeOne.insertPoint(oldTree.dataset(bestIndex));
    else
      treeTwo.insertPoint(oldTree.dataset(bestIndex));

    oldTree.dataset.col(bestIndex) = oldTree.dataset.col(--end); // decrement end.
  }

  // See if we need to satisfy the minimum fill.
  if(end > 1) {
    if(numAssignedOne < numAssignedTwo) {
      for(int i = 0; i < end; i++) {
        treeOne.insertPoint(oldTree.dataset(i));
      }
    } else {
      for(int i = 0; i < end; i++) {
        treeTwo.insertPoint(oldTree.dataset(i));
      }
    }
  }
}

template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<SplitType, DescentType, StatisticType, MatType>::AssignNodeDestNode(
    const RectangleTree<SplitType, DescentType, StatisticType, MatType>& oldTree,
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& treeOne,
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& treeTwo,
    const int intI,
    const int intJ)
{
  
  int end = oldTree.getNumChildren();
  assert(end > 1); // If this isn't true, the tree is really weird.

  treeOne.getChildren()[0] = oldTree.getChildren()[intI];
  oldTree.getChildren[intI] = oldTree.getChildren()[--end]; // decrement end
  treeTwo.getChildren()[0] = oldTree.getChildren()[intJ];
  oldTree.getChildren()[intJ] = oldTree.getChildren()[--end]; // decrement end
 
  int numAssignTreeOne = 1;
  int numAssignTreeTwo = 1;

  // In each iteration, we go through all of the nodes and find the one that causes the least
  // increase of volume when added to one of the two new rectangles.  We then add it to that
  // rectangle.
  while(end > 1 && end < oldTree.getMinNumChildren() - std::min(numAssignTreeOne, numAssignTreeTwo)) {
    int bestIndex = 0;
    double bestScore = 0;
    int bestRect = 0;

    // Calculate the increase in volume for assigning this node to each of the new rectangles.
    double volOne = 1.0;
    double volTwo = 1.0;
    for(int i = 0; i < oldTree.Bound().Dim(); i++) {
      volOne *= treeOne.bound[i].width();
      volTwo *= treeTwo.bound[i].width();
    }

    for(int index = 0; index < end; index++) {
      double newVolOne = 1.0;
      double newVolTwo = 1.0;
      for(int i = 0; i < oldTree.Bound().Dim(); i++) {
	// For each of the new rectangles, find the width in this dimension if we add the rectangle at index to
	// the new rectangle.
	math::Range range = oldTree.getChildren()[index].Bound(i);
	newVolOne *= treeOne.Bound(i).Contains(range) ? treeOne.bound[i].width() :
	  (range.Contains(treeOne.Bound(i)) ? range.Width() : (range.lo() < treeOne.Bound(i).lo() ? (treeOne.Bound(i).hi() - range.lo()) : 
		(range.hi() - treeOne.Bound(i).lo())));
	newVolTwo *= treeTwo.Bound(i).Contains(range) ? treeTwo.bound[i].width() :
	  (range.Contains(treeTwo.Bound(i)) ? range.Width() : (range.lo() < treeTwo.Bound(i).lo() ? (treeTwo.Bound(i).hi() - range.lo()) : 
		(range.hi() - treeTwo.Bound(i).lo())));
      }
    
      // Choose the rectangle that requires the lesser increase in volume.
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

    // Assign the rectangle that causes the least increase in volume 
    // to the appropriate rectangle.
    if(bestRect == 1)
      insertNodeIntoTree(treeOne, oldTree.Children()[bestIndex]);
    else
      insertNodeIntoTree(treeTwo, oldTree.Children()[bestIndex]);

    oldTree.Children()[bestIndex] = oldTree.Children()[--end]; // Decrement end.
  }
  // See if we need to satisfy the minimum fill.
  if(end > 1) {
    if(numAssignTreeOne < numAssignTreeTwo) {
      for(int i = 0; i < end; i++) {
        insertNodeIntoTree(treeOne, oldTree.Children()[i]);
      }
    } else {
      for(int i = 0; i < end; i++) {
        insertNodeIntoTree(treeTwo, oldTree.Children()[i]);
      }
    }
  }
}

/**
  * Insert a node into another node.  Expanding the bounds and updating the numberOfChildren.
  */
template<typename SplitType,
	 typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<SplitType, DescentType, StatisticType, MatType>::insertNodeIntoTree(
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& destTree,
    RectangleTree<SplitType, DescentType, StatisticType, MatType>& srcNode)
{
  destTree.Bound() |= srcNode.Bound();
  destTree.Children()[destTree.getNumOfChildren()++] = &srcNode;
}


}; // namespace tree
}; // namespace mlpack

#endif
