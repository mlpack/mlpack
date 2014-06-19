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
template<typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::SplitLeafNode(
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>* tree)
{
  // Use the quadratic split method from: Guttman "R-Trees: A Dynamic Index Structure for
  // Spatial Searching"  It is simplified since we don't handle rectangles, only points.
  // It assumes that the tree uses Euclidean Distance.
  int i = 0;
  int j = 0;
  GetPointSeeds(*tree, &i, &j);

  // If we are splitting the root node, we need will do things differently so that the constructor
  // and other methods don't confuse the end user by giving an address of another node.
  if(tree->Parent() == NULL) {
    
    return;
  }
  
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType> *treeOne = new 
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>(*(tree->Parent()));
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType> *treeTwo = new 
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>(*(tree->Parent()));
  
  // This will assign the ith and jth point appropriately.
  AssignPointDestNode(tree, treeOne, treeTwo, i, j);
  
  //Remove this node and insert treeOne and treeTwo
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>* par = tree->Parent();
  int index = 0;
  for(int i = 0; i < par->NumChildren(); i++) {
    if(par->Children()[i] == tree) {
      index = i;
      break;
    }
  }
  par->Children()[index] = treeOne;
  par->Children()[par->NumChildren()++] = treeTwo;

  //because we copied the points to treeOne and treeTwo, we can just delete this node
  delete tree;

  // we only add one at a time, so we should only need to test for equality
  // just in case, we use an assert.
  assert(par->NumChildren() <= par->MaxNumChildren());

  if(par->NumChildren() == par->MaxNumChildren()) {
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
template<typename DescentType,
	 typename StatisticType,
	 typename MatType>
bool RTreeSplit<DescentType, StatisticType, MatType>::SplitNonLeafNode(
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>* tree)
{
  int i = 0;
  int j = 0;
  GetBoundSeeds(*tree, &i, &j);
  
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>* treeOne = new 
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>(*(tree->Parent()));
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>* treeTwo = new 
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>(*(tree->Parent()));
  
  // This will assign the ith and jth rectangles appropriately.
  AssignNodeDestNode(tree, treeOne, treeTwo, i, j);

  // If we are splitting the root node, we need will do things differently so that the constructor
  // and other methods don't confuse the end user by giving an address of another node.
  if(tree->Parent() == NULL) {
    //tree->Parent() = new RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>(NULL);
    
    return true;
  }
  
  //Remove this node and insert treeOne and treeTwo
  RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>* par = tree->Parent();
  int index = 0;
  for(int i = 0; i < par->NumChildren(); i++) {
    if(par->Children()[i] == tree) {
      index = i;
      break;
    }
  }
  par->Children()[index] = treeOne;
  par->Children()[par->NumChildren()++] = treeTwo;

  // Because we now have pointers to the information stored under this tree,
  // we need to delete this node carefully.
  tree->softDelete();

  // we only add one at a time, so should only need to test for equality
  // just in case, we use an assert.
  assert(par->NumChildren() <= par->MaxNumChildren());

  if(par->NumChildren() == par->MaxNumChildren()) {
    SplitNonLeafNode(par);
  }
  return false;
}

/**
 * Get the two points that will be used as seeds for the split of a leaf node.
 * The indices of these points will be stored in iRet and jRet.
 */
template<typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::GetPointSeeds(
  const RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>& tree,
  int* iRet,
  int* jRet)
{
  // Here we want to find the pair of points that it is worst to place in the same
  // node.  Because we are just using points, we will simply choose the two that would
  // create the most voluminous hyperrectangle.
  double worstPairScore = 0.0;
  int worstI = 0;
  int worstJ = 0;
  for(int i = 0; i < tree.Count(); i++) {
    for(int j = i+1; j < tree.Count(); j++) {
      double score = 1.0;
      for(int k = 0; k < tree.Bound().Dim(); k++) {
	score *= std::abs(tree.Dataset().at(k, i) - tree.Dataset().at(k, j)); // Points are stored by column, but this function takes (row, col).
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
template<typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::GetBoundSeeds(
  const RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType,  StatisticType, MatType>& tree,
  int* iRet,
  int* jRet)
{
  double worstPairScore = 0.0;
  int worstI = 0;
  int worstJ = 0;
  for(int i = 0; i < tree.NumChildren(); i++) {
    for(int j = i+1; j < tree.NumChildren(); j++) {
      double score = 1.0;
      for(int k = 0; k < tree.Bound().Dim(); k++) {
	score *= std::max(tree.Children()[i]->Bound()[k].Hi(), tree.Children()[j]->Bound()[k].Hi()) - 
	  std::min(tree.Children()[i]->Bound()[k].Lo(), tree.Children()[j]->Bound()[k].Lo());
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

template<typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::AssignPointDestNode(
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* oldTree,
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* treeOne,
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* treeTwo,
    const int intI,
    const int intJ)
{
  int end = oldTree->Count();
  assert(end > 1); // If this isn't true, the tree is really weird.

  treeOne->InsertPoint(oldTree->Dataset().col(intI));
  oldTree->Dataset().col(intI) = oldTree->Dataset().col(--end); // decrement end
  treeTwo->InsertPoint(oldTree->Dataset().col(intJ));
  oldTree->Dataset().col(intJ) = oldTree->Dataset().col(--end); // decrement end
  

  int numAssignedOne = 1;
  int numAssignedTwo = 1;

  // In each iteration, we go through all points and find the one that causes the least
  // increase of volume when added to one of the rectangles.  We then add it to that
  // rectangle.  We stop when we run out of points or when all of the remaining points
  // need to be assigned to the same rectangle to satisfy the minimum fill requirement.
  
  // The below is safe because if end decreases and the right hand side of the second part of the conjunction changes
  // on the same iteration, we added the point to the node with fewer points anyways.
  while(end > 1 && end < oldTree->MinLeafSize() - std::min(numAssignedOne, numAssignedTwo)) {
    int bestIndex = 0;
    double bestScore = DBL_MAX;
    int bestRect = 1;

    // Calculate the increase in volume for assigning this point to each rectangle.
    
    // First, calculate the starting volume.
    double volOne = 1.0;
    double volTwo = 1.0;
    for(int i = 0; i < oldTree->Bound().Dim(); i++) {
      volOne *= treeOne->Bound()[i].Width();
      volTwo *= treeTwo->Bound()[i].Width();
    }

    // Find the point that, when assigned to one of the two new rectangles, minimizes the increase
    // in volume.
    for(int index = 0; index < end; index++) {
      double newVolOne = 1.0;
      double newVolTwo = 1.0;
      for(int i = 0; i < oldTree->Bound().Dim(); i++) {
	double c = oldTree->Dataset().col(index)[i];      
	newVolOne *= treeOne->Bound()[i].Contains(c) ? treeOne->Bound()[i].Width() :
	  (c < treeOne->Bound()[i].Lo() ? (treeOne->Bound()[i].Hi() - c) : (c - treeOne->Bound()[i].Lo()));
	newVolTwo *= treeTwo->Bound()[i].Contains(c) ? treeTwo->Bound()[i].Width() :
	  (c < treeTwo->Bound()[i].Lo() ? (treeTwo->Bound()[i].Hi() - c) : (c - treeTwo->Bound()[i].Lo()));
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
      treeOne->InsertPoint(oldTree->Dataset().col(bestIndex));
    else
      treeTwo->InsertPoint(oldTree->Dataset().col(bestIndex));

    oldTree->Dataset().col(bestIndex) = oldTree->Dataset().col(--end); // decrement end.
  }

  // See if we need to satisfy the minimum fill.
  if(end > 1) {
    if(numAssignedOne < numAssignedTwo) {
      for(int i = 0; i < end; i++) {
        treeOne->InsertPoint(oldTree->Dataset().col(i));
      }
    } else {
      for(int i = 0; i < end; i++) {
        treeTwo->InsertPoint(oldTree->Dataset().col(i));
      }
    }
  }
}

template<typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::AssignNodeDestNode(
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* oldTree,
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* treeOne,
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* treeTwo,
    const int intI,
    const int intJ)
{
  
  int end = oldTree->NumChildren();
  assert(end > 1); // If this isn't true, the tree is really weird.

  treeOne->Children()[0] = oldTree->Children()[intI];
  oldTree->Children()[intI] = oldTree->Children()[--end]; // decrement end
  treeTwo->Children()[0] = oldTree->Children()[intJ];
  oldTree->Children()[intJ] = oldTree->Children()[--end]; // decrement end
 
  int numAssignTreeOne = 1;
  int numAssignTreeTwo = 1;

  // In each iteration, we go through all of the nodes and find the one that causes the least
  // increase of volume when added to one of the two new rectangles.  We then add it to that
  // rectangle.
  while(end > 1 && end < oldTree->MinNumChildren() - std::min(numAssignTreeOne, numAssignTreeTwo)) {
    int bestIndex = 0;
    double bestScore = DBL_MAX;
    int bestRect = 0;

    // Calculate the increase in volume for assigning this node to each of the new rectangles.
    double volOne = 1.0;
    double volTwo = 1.0;
    for(int i = 0; i < oldTree->Bound().Dim(); i++) {
      volOne *= treeOne->Bound()[i].Width();
      volTwo *= treeTwo->Bound()[i].Width();
    }

    for(int index = 0; index < end; index++) {
      double newVolOne = 1.0;
      double newVolTwo = 1.0;
      for(int i = 0; i < oldTree->Bound().Dim(); i++) {
	// For each of the new rectangles, find the width in this dimension if we add the rectangle at index to
	// the new rectangle.
	math::Range range = oldTree->Children()[index]->Bound()[i];
	newVolOne *= treeOne->Bound()[i].Contains(range) ? treeOne->Bound()[i].Width() :
	  (range.Contains(treeOne->Bound()[i]) ? range.Width() : (range.Lo() < treeOne->Bound()[i].Lo() ? (treeOne->Bound()[i].Hi() - range.Lo()) : 
		(range.Hi() - treeOne->Bound()[i].Lo())));
	newVolTwo *= treeTwo->Bound()[i].Contains(range) ? treeTwo->Bound()[i].Width() :
	  (range.Contains(treeTwo->Bound()[i]) ? range.Width() : (range.Lo() < treeTwo->Bound()[i].Lo() ? (treeTwo->Bound()[i].Hi() - range.Lo()) : 
		(range.Hi() - treeTwo->Bound()[i].Lo())));
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
      insertNodeIntoTree(treeOne, oldTree->Children()[bestIndex]);
    else
      insertNodeIntoTree(treeTwo, oldTree->Children()[bestIndex]);

    oldTree->Children()[bestIndex] = oldTree->Children()[--end]; // Decrement end.
  }
  // See if we need to satisfy the minimum fill.
  if(end > 1) {
    if(numAssignTreeOne < numAssignTreeTwo) {
      for(int i = 0; i < end; i++) {
        insertNodeIntoTree(treeOne, oldTree->Children()[i]);
      }
    } else {
      for(int i = 0; i < end; i++) {
        insertNodeIntoTree(treeTwo, oldTree->Children()[i]);
      }
    }
  }
}

/**
  * Insert a node into another node.  Expanding the bounds and updating the numberOfChildren.
  */
template<typename DescentType,
	 typename StatisticType,
	 typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::insertNodeIntoTree(
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* destTree,
    RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* srcNode)
{
  destTree->Bound() |= srcNode->Bound();
  destTree->Children()[destTree->NumChildren()++] = srcNode;
}


}; // namespace tree
}; // namespace mlpack

#endif
