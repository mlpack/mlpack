/**
 * @file neighbor_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Neighbor-Search class to perform all-nearest-neighbors on
 * two specified data sets.
 *
 * This file is part of MLPACK 1.0.6.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

#include "neighbor_search_rules.hpp"

using namespace mlpack::neighbor;

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::
NeighborSearch(const typename TreeType::Mat& referenceSet,
               const typename TreeType::Mat& querySet,
               const bool naive,
               const bool singleMode,
               const size_t leafSize,
               const MetricType metric) :
    referenceCopy(referenceSet),
    queryCopy(querySet),
    referenceSet(referenceCopy),
    querySet(queryCopy),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(true), // False if a tree was passed.
    hasQuerySet(true),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // C++11 will allow us to call out to other constructors so we can avoid this
  // copypasta problem.

  // We'll time tree building, but only if we are building trees.
  Timer::Start("tree_building");

  // Construct as a naive object if we need to.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceCopy.n_cols : leafSize));

  if (!singleMode)
    queryTree = new TreeType(queryCopy, oldFromNewQueries,
        (naive ? querySet.n_cols : leafSize));

  // Stop the timer we started above (if we need to).
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::
NeighborSearch(const typename TreeType::Mat& referenceSet,
               const bool naive,
               const bool singleMode,
               const size_t leafSize,
               const MetricType metric) :
    referenceCopy(referenceSet),
    referenceSet(referenceCopy),
    querySet(referenceCopy),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(true),
    hasQuerySet(false),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // We'll time tree building, but only if we are building trees.
  Timer::Start("tree_building");

  // Construct as a naive object if we need to.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceSet.n_cols : leafSize));
  if (!singleMode)
    queryTree = new TreeType(*referenceTree);

  // Stop the timer we started above.
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::NeighborSearch(
    TreeType* referenceTree,
    TreeType* queryTree,
    const typename TreeType::Mat& referenceSet,
    const typename TreeType::Mat& querySet,
    const bool singleMode,
    const MetricType metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    referenceTree(referenceTree),
    queryTree(queryTree),
    treeOwner(false),
    hasQuerySet(true),
    naive(false),
    singleMode(singleMode),
    metric(metric),
    numberOfPrunes(0)
{
  // Nothing else to initialize.
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::NeighborSearch(
    TreeType* referenceTree,
    const typename TreeType::Mat& referenceSet,
    const bool singleMode,
    const MetricType metric) :
    referenceSet(referenceSet),
    querySet(referenceSet),
    referenceTree(referenceTree),
    queryTree(NULL),
    treeOwner(false),
    hasQuerySet(false), // In this case we will own a tree, if singleMode.
    naive(false),
    singleMode(singleMode),
    metric(metric),
    numberOfPrunes(0)
{
  Timer::Start("tree_building");

  // The query tree cannot be the same as the reference tree.
  if (referenceTree && !singleMode)
    queryTree = new TreeType(*referenceTree);

  Timer::Stop("tree_building");
}

/**
 * The tree is the only member we may be responsible for deleting.  The others
 * will take care of themselves.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
NeighborSearch<SortPolicy, MetricType, TreeType>::~NeighborSearch()
{
  if (treeOwner)
  {
    if (referenceTree)
      delete referenceTree;
    if (queryTree)
      delete queryTree;
  }
  else if (!treeOwner && !hasQuerySet && !singleMode)
  {
    // We replicated the reference tree to create a query tree.
    delete queryTree;
  }
}

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void NeighborSearch<SortPolicy, MetricType, TreeType>::Search(
    const size_t k,
    arma::Mat<size_t>& resultingNeighbors,
    arma::mat& distances)
{
  Timer::Start("computing_neighbors");

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid an extra copy, we will store the neighbors and distances in a
  // separate matrix.
  arma::Mat<size_t>* neighborPtr = &resultingNeighbors;
  arma::mat* distancePtr = &distances;

  if (treeOwner && !(singleMode && hasQuerySet))
    distancePtr = new arma::mat; // Query indices need to be mapped.
  if (treeOwner)
    neighborPtr = new arma::Mat<size_t>; // All indices need mapping.

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  size_t numPrunes = 0;

  // Create the helper object for the tree traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr, metric);

  if (singleMode)
  {
    // Create the traverser.
    typename TreeType::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);
  }
  else // Dual-tree recursion.
  {
    // Create the traverser.
    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*queryTree, *referenceTree);

    Log::Info << traverser.NumVisited() << " node combinations were visited.\n";
    Log::Info << traverser.NumScores() << " node combinations were scored.\n";
    Log::Info << traverser.NumBaseCases() << " base cases were calculated.\n";
  }

  Timer::Stop("computing_neighbors");

  // Now, do we need to do mapping of indices?
  if (!treeOwner)
  {
    // No mapping needed.  We are done.
    return;
  }
  else if (treeOwner && hasQuerySet && !singleMode) // Map both sets.
  {
    // Set size of output matrices correctly.
    resultingNeighbors.set_size(k, querySet.n_cols);
    distances.set_size(k, querySet.n_cols);

    for (size_t i = 0; i < distances.n_cols; i++)
    {
      // Map distances (copy a column).
      distances.col(oldFromNewQueries[i]) = distancePtr->col(i);

      // Map indices of neighbors.
      for (size_t j = 0; j < distances.n_rows; j++)
      {
        resultingNeighbors(j, oldFromNewQueries[i]) =
            oldFromNewReferences[(*neighborPtr)(j, i)];
      }
    }

    // Finished with temporary matrices.
    delete neighborPtr;
    delete distancePtr;
  }
  else if (treeOwner && !hasQuerySet)
  {
    resultingNeighbors.set_size(k, querySet.n_cols);
    distances.set_size(k, querySet.n_cols);

    for (size_t i = 0; i < distances.n_cols; i++)
    {
      // Map distances (copy a column).
      distances.col(oldFromNewReferences[i]) = distancePtr->col(i);

      // Map indices of neighbors.
      for (size_t j = 0; j < distances.n_rows; j++)
      {
        resultingNeighbors(j, oldFromNewReferences[i]) =
            oldFromNewReferences[(*neighborPtr)(j, i)];
      }
    }
  }
  else if (treeOwner && hasQuerySet && singleMode) // Map only references.
  {
    // Set size of neighbor indices matrix correctly.
    resultingNeighbors.set_size(k, querySet.n_cols);

    // Map indices of neighbors.
    for (size_t i = 0; i < resultingNeighbors.n_cols; i++)
    {
      for (size_t j = 0; j < resultingNeighbors.n_rows; j++)
      {
        resultingNeighbors(j, i) = oldFromNewReferences[(*neighborPtr)(j, i)];
      }
    }

    // Finished with temporary matrix.
    delete neighborPtr;
  }
} // Search

#endif
