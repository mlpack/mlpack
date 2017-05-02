/**
 * @file r_star_tree_split_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of class (RStarTreeSplit) to split a RectangleTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_IMPL_HPP

#include "r_star_tree_split.hpp"
#include "rectangle_tree.hpp"

#include <mlpack/core/math/range.hpp>

namespace mlpack {
namespace tree {

/**
 * We call GetPointSeeds to get the two points which will be the initial points
 * in the new nodes We then call AssignPointDestNode to assign the remaining
 * points to the two new nodes.  Finally, we delete the old node and insert the
 * new nodes into the tree, spliting the parent if necessary.
 */
template<typename TreeType>
void RStarTreeSplit::SplitLeafNode(TreeType *tree,std::vector<bool>& relevels)
{
  // Convenience typedef.
  typedef typename TreeType::ElemType ElemType;
  typedef bound::HRectBound<metric::EuclideanDistance, ElemType> BoundType;

  if (tree->Count() <= tree->MaxLeafSize())
    return;
//  std::cout << "split leaf node " << tree << " with parent " << tree->Parent()
//<< "\n";

  // If we are splitting the root node, we need will do things differently so
  // that the constructor and other methods don't confuse the end user by giving
  // an address of another node.
//  if (tree->Parent() == NULL)
//  {
    // We actually want to copy this way.  Pointers and everything.
//    TreeType* copy = new TreeType(*tree, false);
//    copy->Parent() = tree;
//    tree->Count() = 0;
//    tree->NullifyData();
    // Because this was a leaf node, numChildren must be 0.
//    tree->children[(tree->NumChildren())++] = copy;
//    assert(tree->NumChildren() == 1);

//    RStarTreeSplit::SplitLeafNode(copy,relevels);
//    return;
//  }

  // If we haven't yet reinserted on this level, we try doing so now.
  if (relevels[tree->TreeDepth() - 1])
  {
    relevels[tree->TreeDepth() - 1] = false;

    // We sort the points by decreasing distance to the centroid of the bound.
    // We then remove the first p entries and reinsert them at the root.
    TreeType* root = tree;
    while (root->Parent() != NULL)
      root = root->Parent();
    size_t p = tree->MaxLeafSize() * 0.3; // The paper says this works the best.
    if (p == 0)
    {
      RStarTreeSplit::SplitLeafNode(tree,relevels);
      return;
    }

    std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
    arma::Col<ElemType> center;
    tree->Bound().Center(center); // Modifies centroid.
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Metric().Evaluate(center,
          tree->Dataset().col(tree->Point(i)));
      sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, size_t>);
    std::vector<size_t> pointIndices(sorted.size());

    for (size_t i = 0; i < p; i++)
    {
      // We start from the end of sorted.
      pointIndices[i] = tree->Point(sorted[sorted.size() - 1 - i].second);

      root->DeletePoint(tree->Point(sorted[sorted.size() - 1 - i].second),
          relevels);
    }

    for (size_t i = 0; i < p; i++)
    {
      // We reverse the order again to reinsert the closest points first.
      root->InsertPoint(pointIndices[p - 1 - i], relevels);
    }

    return;
  }

  int bestOverlapIndexOnBestAxis = 0;
  int bestAreaIndexOnBestAxis = 0;
  bool tiedOnOverlap = false;
  int bestAxis = 0;
  ElemType bestAxisScore = std::numeric_limits<ElemType>::max();

  for (size_t j = 0; j < tree->Bound().Dim(); j++)
  {
    ElemType axisScore = 0.0;
    // Since we only have points in the leaf nodes, we only need to sort once.
    std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Dataset().col(tree->Point(i))[j];
      sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, size_t>);

    // We'll store each of the three scores for each distribution.
    std::vector<ElemType> areas(tree->MaxLeafSize() - 2 * tree->MinLeafSize() +
        2);
    std::vector<ElemType> margins(tree->MaxLeafSize() - 2 * tree->MinLeafSize() +
        2);
    std::vector<ElemType> overlapedAreas(tree->MaxLeafSize() -
        2 * tree->MinLeafSize() + 2);

    for (size_t i = 0; i < areas.size(); i++)
    {
      areas[i] = 0.0;
      margins[i] = 0.0;
      overlapedAreas[i] = 0.0;
    }

    for (size_t i = 0; i < areas.size(); i++)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinLeafSize() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.

      size_t cutOff = tree->MinLeafSize() + i;

      BoundType bound1(tree->Bound().Dim());
      BoundType bound2(tree->Bound().Dim());

      for (size_t l = 0; l < cutOff; l++)
        bound1 |= tree->Dataset().col(tree->Point(sorted[l].second));

      for (size_t l = cutOff; l < tree->Count(); l++)
        bound2 |= tree->Dataset().col(tree->Point(sorted[l].second));

      ElemType area1 = bound1.Volume();
      ElemType area2 = bound2.Volume();
      ElemType oArea = bound1.Overlap(bound2);

      for (size_t k = 0; k < bound1.Dim(); k++)
        margins[i] += bound1[k].Width() + bound2[k].Width();

      areas[i] += area1 + area2;
      overlapedAreas[i] += oArea;
      axisScore += margins[i];
    }

    if (axisScore < bestAxisScore)
    {
      bestAxisScore = axisScore;
      bestAxis = j;
      bestOverlapIndexOnBestAxis = 0;
      bestAreaIndexOnBestAxis = 0;

      for (size_t i = 1; i < areas.size(); i++)
      {
        if (overlapedAreas[i] < overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = false;
          bestAreaIndexOnBestAxis = i;
          bestOverlapIndexOnBestAxis = i;
        }
        else if (overlapedAreas[i] ==
            overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = true;
          if (areas[i] < areas[bestAreaIndexOnBestAxis])
            bestAreaIndexOnBestAxis = i;
        }
      }
    }
  }

  std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
  for (size_t i = 0; i < sorted.size(); i++)
  {
    sorted[i].first = tree->Dataset().col(tree->Point(i))[bestAxis];
    sorted[i].second = tree->Point(i);
  }

  std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, size_t>);

  if (tree->Parent())
  {
//    std::cout << "first check parent " << tree->Parent() << " numDescendants "
//<< tree->Parent()->NumDescendants() <<
//" with " << tree->Parent()->NumChildren() << " children\n";
//    size_t manualCount = 0;
//    for (size_t i = 0; i < tree->Parent()->NumChildren(); ++i)
//    {
//      std::cout << "(" << &(tree->Parent()->Child(i)) << ") ";
//      manualCount += tree->Parent()->Child(i).NumDescendants();
//      std::cout << tree->Parent()->Child(i).NumDescendants() << " ";
//    }
//    std::cout << "\n";
  //  TreeType* treeOne = new TreeType(tree->Parent());
    // Now clean the node, and we will re-use this.
    const size_t oldDescendants = tree->numDescendants;
    const size_t numPoints = tree->count;
    tree->numChildren = 0;
    tree->numDescendants = 0;
    tree->bound.Clear();
    tree->count = 0;
    tree->begin = 0;

    TreeType* treeTwo = new TreeType(tree->Parent());

    if (tiedOnOverlap)
    {
      for (size_t i = 0; i < numPoints; i++)
      {
        if (i < bestAreaIndexOnBestAxis + tree->MinLeafSize())
          tree->InsertPoint(sorted[i].second);
        else
          treeTwo->InsertPoint(sorted[i].second);
      }
    }
    else
    {
      for (size_t i = 0; i < numPoints; i++)
      {
        if (i < bestOverlapIndexOnBestAxis + tree->MinLeafSize())
          tree->InsertPoint(sorted[i].second);
        else
          treeTwo->InsertPoint(sorted[i].second);
      }
    }

    // Insert the new tree node.
    TreeType* par = tree->Parent();
    par->children[par->NumChildren()++] = treeTwo;

    // We only add one at a time, so we should only need to test for equality
    // just in case, we use an assert.
//    std::cout << "x: " << oldDescendants << " == " << tree->NumDescendants() <<
//" + " << treeTwo->NumDescendants() << " (" << tree->NumDescendants() +
//treeTwo->NumDescendants() << ")\n";
    assert(oldDescendants == tree->NumDescendants() + treeTwo->NumDescendants());
//    std::cout << "check parent " << par << " numDescendants " << par->NumDescendants() <<
//" with " << par->NumChildren() << " children\n";
//    manualCount = 0;
//    for (size_t i = 0; i < par->NumChildren(); ++i)
//    {
//      std::cout << "(" << &(par->Child(i)) << ") ";
//      manualCount += par->Child(i).NumDescendants();
//      std::cout << par->Child(i).NumDescendants() << " ";
//    }
//    std::cout << "\n";
//    assert(par->NumDescendants() == manualCount);
    assert(par->NumChildren() <= par->MaxNumChildren() + 1);
    if (par->NumChildren() == par->MaxNumChildren() + 1)
      RStarTreeSplit::SplitNonLeafNode(par, relevels);

    assert(tree->Parent()->NumChildren() <= tree->MaxNumChildren());
    assert(tree->Parent()->NumChildren() >= tree->MinNumChildren());
    assert(treeTwo->Parent()->NumChildren() <= treeTwo->MaxNumChildren());
    assert(treeTwo->Parent()->NumChildren() >= treeTwo->MinNumChildren());
  }
  else
  {
    TreeType* treeOne = new TreeType(tree);
    TreeType* treeTwo = new TreeType(tree);

    const size_t oldNumDescendants = tree->NumDescendants();
    const size_t numPoints = tree->Count();
    tree->numChildren = 0;
    tree->bound.Clear();
    tree->count = 0;
    tree->begin = 0;
    tree->numDescendants = 0;

    if (tiedOnOverlap)
    {
      for (size_t i = 0; i < numPoints; ++i)
      {
        if (i < bestAreaIndexOnBestAxis + tree->MinLeafSize())
          treeOne->InsertPoint(sorted[i].second);
        else
          treeTwo->InsertPoint(sorted[i].second);
      }
    }
    else
    {
      for (size_t i = 0; i < numPoints; ++i)
      {
        if (i < bestOverlapIndexOnBestAxis + tree->MinLeafSize())
          treeOne->InsertPoint(sorted[i].second);
        else
          treeTwo->InsertPoint(sorted[i].second);
      }
    }

    InsertNodeIntoTree(tree, treeOne);
    InsertNodeIntoTree(tree, treeTwo);

//    std::cout << "y: " << oldNumDescendants << ": " << treeOne->NumDescendants()
//<< " + " << treeTwo->NumDescendants() << " (" << tree->NumDescendants() <<
//")\n";
//    std::cout << "tree left " << treeOne->NumDescendants() << ", right " <<
//treeTwo->NumDescendants() << ", parent " << tree->NumDescendants() << "\n";
//    for (size_t i = 0; i < tree->NumChildren(); ++i)
//      std::cout << tree->Child(i).NumDescendants() << " ";
//    std::cout << " (that's the children of the parent)\n";
    assert(oldNumDescendants == tree->NumDescendants());
  }
}

/**
 * We call GetBoundSeeds to get the two new nodes that this one will be broken
 * into.  Then we call AssignNodeDestNode to move the children of this node into
 * either of those two nodes.  Finally, we delete the now unused information and
 * recurse up the tree if necessary.  We don't need to worry about the bounds
 * higher up the tree because they were already updated if necessary.
 */
template<typename TreeType>
bool RStarTreeSplit::SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels)
{
//  std::cout << "split nonleaf node " << tree << " with parent " <<
//tree->Parent() << "\n";
  // Convenience typedef.
  typedef typename TreeType::ElemType ElemType;
  typedef bound::HRectBound<metric::EuclideanDistance, ElemType> BoundType;

  // If we are splitting the root node, we need will do things differently so
  // that the constructor and other methods don't confuse the end user by giving
  // an address of another node.
//  if (tree->Parent() == NULL)
//  {
    // We actually want to copy this way.  Pointers and everything.
//    TreeType* copy = new TreeType(*tree, false);

//    copy->Parent() = tree;
//    tree->NumChildren() = 0;
//    tree->NullifyData();
//    tree->children[(tree->NumChildren())++] = copy;

//    RStarTreeSplit::SplitNonLeafNode(copy,relevels);
//    return true;
//  }

 /*
  // If we haven't yet reinserted on this level, we try doing so now.
  if (relevels[tree->TreeDepth()]) {
    relevels[tree->TreeDepth()] = false;
    // We sort the points by decreasing centroid to centroid distance.
    // We then remove the first p entries and reinsert them at the root.
    RectangleTree<RStarTreeSplit, DescentType, StatisticType, MatType>* root = tree;
    while(root->Parent() != NULL)
      root = root->Parent();
    size_t p = tree->MaxNumChildren() * 0.3; // The paper says this works the best.
    if (p == 0) {
      SplitNonLeafNode(tree, relevels);
      return false;
    }

    std::vector<sortStruct> sorted(tree->NumChildren());
    arma::vec c1;
    tree->Bound().Center(c1); // Modifies c1.
    for(size_t i = 0; i < sorted.size(); i++) {
      arma::vec c2;
      tree->Children()[i]->Bound().Center(c2); // Modifies c2.
      sorted[i].d = tree->Bound().Metric().Evaluate(c1,c2);
      sorted[i].n = i;
    }
    std::sort(sorted.begin(), sorted.end(), structComp);

    //bool startBelowMinFill = tree->NumChildren() < tree->MinNumChildren();

    std::vector<RectangleTree<RStarTreeSplit, DescentType, StatisticType, MatType>*> removedNodes(p);
    for(size_t i =0; i < p; i++) {
      removedNodes[i] = tree->Children()[sorted[sorted.size()-1-i].n]; // We start from the end of sorted.
      root->RemoveNode(tree->Children()[sorted[sorted.size()-1-i].n], relevels);
    }

    for(size_t i = 0; i < p; i++) {
      root->InsertNode(removedNodes[p-1-i], tree->TreeDepth(), relevels); // We reverse the order again to reinsert the closest nodes first.
    }

    // If we went below min fill, delete this node and reinsert all children.
    //SOMETHING IS WRONG.  SHOULD NOT GO BELOW MIN FILL.
//    if (!startBelowMinFill && tree->NumChildren() < tree->MinNumChildren())
//    std::cout<<"MINFILLERROR "<< p << ", " << tree->NumChildren() << "; " << tree->MaxNumChildren()<<std::endl;

//    if (tree->NumChildren() < tree->MinNumChildren()) {
//      std::vector<RectangleTree<RStarTreeSplit, DescentType, StatisticType, MatType>*> rmNodes(tree->NumChildren());
//      for(size_t i = 0; i < rmNodes.size(); i++) {
//        rmNodes[i] = tree->Children()[i];
//      }
//      root->RemoveNode(tree, relevels);
//      for(size_t i = 0; i < rmNodes.size(); i++) {
//        root->InsertNode(rmNodes[i], tree->TreeDepth(), relevels);
//      }
// //      tree->SoftDelete();
//    }
 //    assert(tree->NumChildren() >= tree->MinNumChildren());
 //    assert(tree->NumChildren() <= tree->MaxNumChildren());

   return false;
 }

*/

  int bestOverlapIndexOnBestAxis = 0;
  int bestAreaIndexOnBestAxis = 0;
  bool tiedOnOverlap = false;
  bool lowIsBest = true;
  int bestAxis = 0;
  ElemType bestAxisScore = std::numeric_limits<ElemType>::max();
  for (size_t j = 0; j < tree->Bound().Dim(); j++)
  {
    ElemType axisScore = 0.0;

    // We'll do Bound().Lo() now and use Bound().Hi() later.
    std::vector<std::pair<ElemType, size_t>> sorted(tree->NumChildren());
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[j].Lo();
      sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, size_t>);

    // We'll store each of the three scores for each distribution.
    std::vector<ElemType> areas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> margins(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> overlapedAreas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);

    for (size_t i = 0; i < areas.size(); i++)
    {
      areas[i] = 0.0;
      margins[i] = 0.0;
      overlapedAreas[i] = 0.0;
    }

    for (size_t i = 0; i < areas.size(); i++)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinNumChildren() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.

      size_t cutOff = tree->MinNumChildren() + i;

      BoundType bound1(tree->Bound().Dim());
      BoundType bound2(tree->Bound().Dim());

      for (size_t l = 0; l < cutOff; l++)
        bound1 |= tree->Child(sorted[l].second).Bound();

      for (size_t l = cutOff; l < tree->NumChildren(); l++)
        bound2 |= tree->Child(sorted[l].second).Bound();

      ElemType area1 = bound1.Volume();
      ElemType area2 = bound2.Volume();
      ElemType oArea = bound1.Overlap(bound2);

      for (size_t k = 0; k < bound1.Dim(); k++)
        margins[i] += bound1[k].Width() + bound2[k].Width();

      areas[i] += area1 + area2;
      overlapedAreas[i] += oArea;
      axisScore += margins[i];
    }

    if (axisScore < bestAxisScore)
    {
      bestAxisScore = axisScore;
      bestAxis = j;
      bestOverlapIndexOnBestAxis = 0;
      bestAreaIndexOnBestAxis = 0;
      for (size_t i = 1; i < areas.size(); i++)
      {
        if (overlapedAreas[i] < overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = false;
          bestAreaIndexOnBestAxis = i;
          bestOverlapIndexOnBestAxis = i;
        }
        else if (overlapedAreas[i] ==
            overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = true;
          if (areas[i] < areas[bestAreaIndexOnBestAxis])
            bestAreaIndexOnBestAxis = i;
        }
      }
    }
  }

  // Now we do the same thing using Bound().Hi() and choose the best of the two.
  for (size_t j = 0; j < tree->Bound().Dim(); j++)
  {
    ElemType axisScore = 0.0;

    std::vector<std::pair<ElemType, size_t>> sorted(tree->NumChildren());
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[j].Hi();
      sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, size_t>);

    // We'll store each of the three scores for each distribution.
    std::vector<ElemType> areas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> margins(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> overlapedAreas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);

    for (size_t i = 0; i < areas.size(); i++)
    {
      areas[i] = 0.0;
      margins[i] = 0.0;
      overlapedAreas[i] = 0.0;
    }

    for (size_t i = 0; i < areas.size(); i++)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinNumChildren() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.

      size_t cutOff = tree->MinNumChildren() + i;

      BoundType bound1(tree->Bound().Dim());
      BoundType bound2(tree->Bound().Dim());

      for (size_t l = 0; l < cutOff; l++)
        bound1 |= tree->Child(sorted[l].second).Bound();

      for (size_t l = cutOff; l < tree->NumChildren(); l++)
        bound2 |= tree->Child(sorted[l].second).Bound();

      ElemType area1 = bound1.Volume();
      ElemType area2 = bound2.Volume();
      ElemType oArea = bound1.Overlap(bound2);

      for (size_t k = 0; k < bound1.Dim(); k++)
        margins[i] += bound1[k].Width() + bound2[k].Width();

      areas[i] += area1 + area2;
      overlapedAreas[i] += oArea;
      axisScore += margins[i];
    }

    if (axisScore < bestAxisScore)
    {
      bestAxisScore = axisScore;
      bestAxis = j;
      lowIsBest = false;
      bestOverlapIndexOnBestAxis = 0;
      bestAreaIndexOnBestAxis = 0;

      for (size_t i = 1; i < areas.size(); i++)
      {
        if (overlapedAreas[i] < overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = false;
          bestAreaIndexOnBestAxis = i;
          bestOverlapIndexOnBestAxis = i;
        }
        else if (overlapedAreas[i] ==
            overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = true;
          if (areas[i] < areas[bestAreaIndexOnBestAxis])
            bestAreaIndexOnBestAxis = i;
        }
      }
    }
  }

  std::vector<std::pair<ElemType, TreeType*>> sorted(tree->NumChildren());
  if (lowIsBest)
  {
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[bestAxis].Lo();
      sorted[i].second = &tree->Child(i);
    }
  }
  else
  {
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[bestAxis].Hi();
      sorted[i].second = &tree->Child(i);
    }
  }

  std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, TreeType*>);

  if (tree->Parent() != NULL)
  {
    const size_t oldNumDescendants = tree->NumDescendants();
//    for (size_t i = 0; i < tree->NumChildren(); ++i)
//      std::cout << tree->Child(i).NumDescendants() << " ";
//    std::cout << " (total " << tree->NumDescendants() << ", count " <<
//tree->count << "\n";
    const size_t oldNumChildren = tree->NumChildren();
    tree->numChildren = 0;
    tree->bound.Clear();
    tree->count = 0;
    tree->begin = 0;
    tree->numDescendants = 0;
    TreeType* treeTwo = new TreeType(tree->Parent());

    if (tiedOnOverlap)
    {
      for (size_t i = 0; i < oldNumChildren; i++)
      {
        if (i < bestAreaIndexOnBestAxis + tree->MinNumChildren())
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into tree"
//              << "\n";
          InsertNodeIntoTree(tree, sorted[i].second);
        }
        else
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into treeTwo\n";
          InsertNodeIntoTree(treeTwo, sorted[i].second);
        }
      }
    }
    else
    {
      for (size_t i = 0; i < oldNumChildren; i++)
      {
        if (i < bestOverlapIndexOnBestAxis + tree->MinNumChildren())
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into tree\n";
          InsertNodeIntoTree(tree, sorted[i].second);
        }
        else
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into treeTwo\n";
          InsertNodeIntoTree(treeTwo, sorted[i].second);
        }
      }
    }

    // Insert the new node into the tree.
    TreeType* par = tree->Parent();
    par->children[par->NumChildren()++] = treeTwo;

    // We only add one at a time, so we should only need to test for equality
    // just in case, we use an assert.
    assert(par->NumChildren() <= par->MaxNumChildren() + 1);
    if (par->NumChildren() == par->MaxNumChildren() + 1)
      RStarTreeSplit::SplitNonLeafNode(par,relevels);

    // We have to update the children of each of these new nodes so that they
    // record the correct parent.
    for (size_t i = 0; i < tree->NumChildren(); i++)
      tree->children[i]->Parent() = tree;

    for (size_t i = 0; i < treeTwo->NumChildren(); i++)
      treeTwo->children[i]->Parent() = treeTwo;

//    std::cout << "tree left " << tree->NumDescendants() << ", right " <<
//treeTwo->NumDescendants() << ", parent " << par->NumDescendants() << "\n";
//    for (size_t i = 0; i < par->NumChildren(); ++i)
//      std::cout << par->Child(i).NumDescendants() << " ";
//    std::cout << " (that's the children of the parent)\n";
    assert(oldNumDescendants == (tree->NumDescendants() +
treeTwo->NumDescendants()));


    assert(tree->Parent()->NumChildren() <= tree->MaxNumChildren());
    assert(tree->Parent()->NumChildren() >= tree->MinNumChildren());
    assert(treeTwo->Parent()->NumChildren() <= treeTwo->MaxNumChildren());
    assert(treeTwo->Parent()->NumChildren() >= treeTwo->MinNumChildren());

    assert(tree->MaxNumChildren() < 7);
    assert(treeTwo->MaxNumChildren() < 7);
  }
  else
  {
    const size_t oldDescendants = tree->NumDescendants();
//    for (size_t i = 0; i < tree->NumChildren(); ++i)
//      std::cout << tree->Child(i).NumDescendants() << " ";
//    std::cout << " (total " << tree->NumDescendants() << ", count " <<
//tree->count << "\n";
    TreeType* treeOne = new TreeType(tree);
    TreeType* treeTwo = new TreeType(tree);

    const size_t oldNumChildren = tree->NumChildren();
    tree->count = 0;
    tree->numChildren = 0;
    tree->bound.Clear();
    tree->numDescendants = 0;

    if (tiedOnOverlap)
    {
      for (size_t i = 0; i < oldNumChildren; i++)
      {
        if (i < bestAreaIndexOnBestAxis + tree->MinNumChildren())
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into treeOne\n";
          InsertNodeIntoTree(treeOne, sorted[i].second);
        }
        else
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into treeTwo\n";
          InsertNodeIntoTree(treeTwo, sorted[i].second);
        }
      }
    }
    else
    {
      for (size_t i = 0; i < oldNumChildren; i++)
      {
        if (i < bestOverlapIndexOnBestAxis + tree->MinNumChildren())
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into treeOne\n";
          InsertNodeIntoTree(treeOne, sorted[i].second);
        }
        else
        {
//          std::cout << "insert " << sorted[i].second->NumDescendants() << " into treeTwo\n";
          InsertNodeIntoTree(treeTwo, sorted[i].second);
        }
      }
    }

    InsertNodeIntoTree(tree, treeOne);
    InsertNodeIntoTree(tree, treeTwo);

//    std::cout << oldDescendants << "; " << treeOne->numDescendants << ", " <<
//treeTwo->numDescendants << " --> " << tree->numDescendants << "\n";
    assert(oldDescendants == (treeOne->numDescendants +
        treeTwo->numDescendants));

    // We have to update the children of each of these new nodes so that they
    // record the correct parent.
    for (size_t i = 0; i < treeOne->NumChildren(); i++)
      treeOne->children[i]->Parent() = treeOne;

    for (size_t i = 0; i < treeTwo->NumChildren(); i++)
      treeTwo->children[i]->Parent() = treeTwo;
  }

  return false;
}

/**
 * Insert a node into another node.  Expanding the bounds and updating the
 * numberOfChildren.
 */
template<typename TreeType>
void RStarTreeSplit::InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode)
{
//  std::cout << "insert " << srcNode << " into " << destTree << "\n";

  destTree->Bound() |= srcNode->Bound();
  destTree->numDescendants += srcNode->numDescendants;
  destTree->children[destTree->NumChildren()++] = srcNode;

//  std::cout << "dest now has " << destTree->NumDescendants() << "\n";
//  size_t manualCount = 0;
//  for (size_t i = 0; i < destTree->NumChildren(); ++i)
//  {
//    manualCount += destTree->Child(i).NumDescendants();
//    std::cout << destTree->Child(i).NumDescendants() << " ";
//  }
//  std::cout << "\n";
//  assert(manualCount == destTree->NumDescendants());
}

} // namespace tree
} // namespace mlpack

#endif
