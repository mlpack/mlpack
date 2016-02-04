/**
 * @file r_star_tree_descent_heuristic_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of RStarTreeDescentHeuristic, a class that chooses the best child of a node in
 * an R tree when inserting a new point.
 *
 * This file is part of mlpack 2.0.1.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_DESCENT_HEURISTIC_IMPL_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_DESCENT_HEURISTIC_IMPL_HPP

#include "r_star_tree_descent_heuristic.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
inline size_t RStarTreeDescentHeuristic::ChooseDescentNode(
    const TreeType* node,
    const arma::vec& point)
{
  bool tiedOne = false;
  std::vector<double> originalScores(node->NumChildren());
  double origMinScore = DBL_MAX;

  if (node->Children()[0]->IsLeaf())
  {
    // If its children are leaf nodes, use minimum overlap to choose.
    double bestIndex = 0;

    for (size_t i = 0; i < node->NumChildren(); i++)
    {
      double sc = 0;
      for (size_t j = 0; j < node->NumChildren(); j++)
      {
        if (j != i)
        {
          double overlap = 1.0;
          double newOverlap = 1.0;
          for (size_t k = 0; k < node->Bound().Dim(); k++)
          {
            double newHigh = std::max(point[k],
                node->Children()[i]->Bound()[k].Hi());
            double newLow = std::min(point[k],
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
  std::vector<double> scores(node->NumChildren());
  if (tiedOne)
  {
    // If the first heuristic was tied, we need to eliminate garbage values.
    for (size_t i = 0; i < scores.size(); i++)
      scores[i] = DBL_MAX;
  }

  std::vector<int> vols(node->NumChildren());
  double minScore = DBL_MAX;
  int bestIndex = 0;
  bool tied = false;

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    if (!tiedOne || originalScores[i] == origMinScore)
    {
      double v1 = 1.0;
      double v2 = 1.0;
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
    double minVol = DBL_MAX;
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
  std::vector<double> scores(node->NumChildren());
  std::vector<int> vols(node->NumChildren());
  double minScore = DBL_MAX;
  int bestIndex = 0;
  bool tied = false;

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    double v1 = 1.0;
    double v2 = 1.0;
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
    double minVol = DBL_MAX;
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
