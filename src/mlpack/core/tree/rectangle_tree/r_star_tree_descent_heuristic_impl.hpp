/**
 * @file r_star_tree_descent_heuristic_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of RStarTreeDescentHeuristic, a class that chooses the best child of a node in
 * an R tree when inserting a new point.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_star_tree_descent_heuristic.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
inline size_t RStarTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* node,
    const arma::vec& point)
{
  // Convenience typedef.
  typedef typename TreeType::ElemType ElemType;

  bool tiedOne = false;
  std::vector<ElemType> originalScores(node->NumChildren());
  ElemType origMinScore = std::numeric_limits<ElemType>::max();

  if (node->Children()[0]->IsLeaf())
  {
    // If its children are leaf nodes, use minimum overlap to choose.
    size_t bestIndex = 0;

    for (size_t i = 0; i < node->NumChildren(); i++)
    {
      ElemType sc = 0;
      for (size_t j = 0; j < node->NumChildren(); j++)
      {
        if (j != i)
        {
          ElemType overlap = 1.0;
          ElemType newOverlap = 1.0;
          for (size_t k = 0; k < node->Bound().Dim(); k++)
          {
            ElemType newHigh = std::max(point[k],
                node->Children()[i]->Bound()[k].Hi());
            ElemType newLow = std::min(point[k],
                node->Children()[i]->Bound()[k].Lo());
            overlap *= node->Children()[i]->Bound()[k].Hi() < node->Children()[j]->Bound()[k].Lo() || node->Children()[i]->Bound()[k].Lo() > node->Children()[j]->Bound()[k].Hi() ? 0 : std::min(node->Children()[i]->Bound()[k].Hi(), node->Children()[j]->Bound()[k].Hi()) - std::max(node->Children()[i]->Bound()[k].Lo(), node->Children()[j]->Bound()[k].Lo());
            newOverlap *= newHigh < node->Children()[j]->Bound()[k].Lo() || newLow > node->Children()[j]->Bound()[k].Hi() ? 0 : std::min(newHigh, node->Children()[j]->Bound()[k].Hi()) - std::max(newLow, node->Children()[j]->Bound()[k].Lo());
          }
          sc += newOverlap - overlap;
        }
      }

      originalScores[i] = sc;
      if (sc < origMinScore)
      {
        origMinScore = sc;
        bestIndex = i;
      }
      else if (sc == origMinScore)
      {
        tiedOne = true;
      }
    }

    if (!tiedOne)
      return bestIndex;
  }

  // We do this if it is not on the second level or if there was a tie.
  std::vector<ElemType> scores(node->NumChildren());
  if (tiedOne)
  {
    // If the first heuristic was tied, we need to eliminate garbage values.
    for (size_t i = 0; i < scores.size(); i++)
      scores[i] = std::numeric_limits<ElemType>::max();
  }

  std::vector<ElemType> vols(node->NumChildren());
  ElemType minScore = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  bool tied = false;

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    if (!tiedOne || originalScores[i] == origMinScore)
    {
      ElemType v1 = 1.0;
      ElemType v2 = 1.0;
      for (size_t j = 0; j < node->Bound().Dim(); j++)
      {
        v1 *= node->Children()[i]->Bound()[j].Width();
        v2 *= node->Children()[i]->Bound()[j].Contains(point[j]) ? node->Children()[i]->Bound()[j].Width() : (node->Children()[i]->Bound()[j].Hi() < point[j] ? (point[j] - node->Children()[i]->Bound()[j].Lo()) :
            (node->Children()[i]->Bound()[j].Hi() - point[j]));
      }

      assert(v2 - v1 >= 0);
      vols[i] = v1;
      scores[i] = v2 - v1;

      if (v2 - v1 < minScore)
      {
        minScore = v2 - v1;
        bestIndex = i;
      }
      else if (v2 - v1 == minScore)
      {
        tied = true;
      }
    }
  }

  if (tied)
  {
    // We break ties by choosing the smallest bound.
    ElemType minVol = std::numeric_limits<ElemType>::max();
    bestIndex = 0;
    for (size_t i = 0; i < scores.size(); i++)
    {
      if (scores[i] == minScore)
      {
        if (vols[i] < minVol)
        {
          minVol = vols[i];
          bestIndex = i;
        }
      }
    }
  }

  return bestIndex;
}

/**
 * We simplify this to the same code as is used in the regular R tree, since the
 * inserted node should always be above the leaf level.  If the tree is
 * eventually changed to support rectangles, this could be changed to match the
 * above code; however, the paper's explanation for their algorithm seems to
 * indicate the above is more for points than for rectangles.
 */
template<typename TreeType>
inline size_t RStarTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* node,
    const TreeType* insertedNode)
{
  // Convenience typedef.
  typedef typename TreeType::ElemType ElemType;

  std::vector<ElemType> scores(node->NumChildren());
  std::vector<ElemType> vols(node->NumChildren());
  ElemType minScore = std::numeric_limits<ElemType>::max();
  size_t bestIndex = 0;
  bool tied = false;

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    ElemType v1 = 1.0;
    ElemType v2 = 1.0;
    for (size_t j = 0; j < node->Children()[i]->Bound().Dim(); j++)
    {
      v1 *= node->Children()[i]->Bound()[j].Width();
      v2 *= node->Children()[i]->Bound()[j].Contains(insertedNode->Bound()[j]) ? node->Children()[i]->Bound()[j].Width() :
              (insertedNode->Bound()[j].Contains(node->Children()[i]->Bound()[j]) ? insertedNode->Bound()[j].Width() : (insertedNode->Bound()[j].Lo() < node->Children()[i]->Bound()[j].Lo() ? (node->Children()[i]->Bound()[j].Hi() - insertedNode->Bound()[j].Lo()) : (insertedNode->Bound()[j].Hi() - node->Children()[i]->Bound()[j].Lo())));
    }

    assert(v2 - v1 >= 0);
    vols[i] = v1;
    scores[i] = v2 - v1;

    if (v2 - v1 < minScore)
    {
      minScore = v2 - v1;
      bestIndex = i;
    }
    else if (v2 - v1 == minScore)
    {
      tied = true;
    }
  }

  if (tied)
  {
    // We break ties by choosing the smallest bound.
    ElemType minVol = std::numeric_limits<ElemType>::max();
    bestIndex = 0;
    for (size_t i = 0; i < scores.size(); i++)
    {
      if (scores[i] == minScore)
      {
        if (vols[i] < minVol)
        {
          minVol = vols[i];
          bestIndex = i;
        }
      }
    }
  }

  return bestIndex;
}

} // namespace tree
} // namespace mlpack

#endif
