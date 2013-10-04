/**
 * @file ra_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of RASearch class to perform rank-approximate
 * all-nearest-neighbors on two specified data sets.
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
#ifndef __MLPACK_METHODS_RANN_RA_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_RANN_RA_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

#include "ra_search_rules.hpp"

namespace mlpack {
namespace neighbor {

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(const typename TreeType::Mat& referenceSet,
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
    ownReferenceTree(true), // False if a tree was passed.
    ownQueryTree(true), // False if a tree was passed.
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // We'll time tree building.
  Timer::Start("tree_building");

  // Construct as a naive object if we need to.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceCopy.n_cols : leafSize));

  queryTree = new TreeType(queryCopy, oldFromNewQueries,
      (naive ? querySet.n_cols : leafSize));

  // Stop the timer we started above.
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(const typename TreeType::Mat& referenceSet,
         const bool naive,
         const bool singleMode,
         const size_t leafSize,
         const MetricType metric) :
    referenceCopy(referenceSet),
    referenceSet(referenceCopy),
    querySet(referenceCopy),
    referenceTree(NULL),
    queryTree(NULL),
    ownReferenceTree(true),
    ownQueryTree(false), // Since it will be the same as referenceTree.
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // We'll time tree building.
  Timer::Start("tree_building");

  // Construct as a naive object if we need to.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceSet.n_cols : leafSize));

  // Stop the timer we started above.
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(TreeType* referenceTree,
         TreeType* queryTree,
         const typename TreeType::Mat& referenceSet,
         const typename TreeType::Mat& querySet,
         const bool singleMode,
         const MetricType metric) :
    referenceSet(referenceSet),
    querySet(querySet),
    referenceTree(referenceTree),
    queryTree(queryTree),
    ownReferenceTree(false),
    ownQueryTree(false),
    naive(false),
    singleMode(singleMode),
    metric(metric),
    numberOfPrunes(0)
// Nothing else to initialize.
{  }

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(TreeType* referenceTree,
         const typename TreeType::Mat& referenceSet,
         const bool singleMode,
         const MetricType metric) :
  referenceSet(referenceSet),
  querySet(referenceSet),
  referenceTree(referenceTree),
  queryTree(NULL),
  ownReferenceTree(false),
  ownQueryTree(false),
  naive(false),
  singleMode(singleMode),
  metric(metric),
  numberOfPrunes(0)
// Nothing else to initialize.
{ }

/**
 * The tree is the only member we may be responsible for deleting.  The others
 * will take care of themselves.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
~RASearch()
{
  if (ownReferenceTree)
    delete referenceTree;
  if (ownQueryTree)
    delete queryTree;
}

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearch<SortPolicy, MetricType, TreeType>::
Search(const size_t k,
       arma::Mat<size_t>& resultingNeighbors,
       arma::mat& distances,
       const double tau,
       const double alpha,
       const bool sampleAtLeaves,
       const bool firstLeafExact,
       const size_t singleSampleLimit)
{
  Timer::Start("computing_neighbors");

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid an extra copy, we will store the neighbors and distances in a
  // separate matrix.
  arma::Mat<size_t>* neighborPtr = &resultingNeighbors;
  arma::mat* distancePtr = &distances;

  if (!naive) // If naive, no re-mapping required since points are not mapped.
  {
    if (ownQueryTree || (ownReferenceTree && !queryTree))
      distancePtr = new arma::mat; // Query indices need to be mapped.
    if (ownReferenceTree || ownQueryTree)
      neighborPtr = new arma::Mat<size_t>; // All indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  size_t numPrunes = 0;

  if (singleMode || naive)
  {
    // Create the helper object for the tree traversal.  Initialization of
    // RASearchRules already implicitly performs the naive tree traversal.
    typedef RASearchRules<SortPolicy, MetricType, TreeType> RuleType;
    RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr,
                   metric, tau, alpha, naive, sampleAtLeaves, firstLeafExact,
                   singleSampleLimit);

    // If the reference root node is a leaf, then the sampling has already been
    // done in the RASearchRules constructor.  This happens when naive = true.
    if (!referenceTree->IsLeaf())
    {
      Log::Info << "Performing single-tree traversal..." << std::endl;

      // Create the traverser.
      typename TreeType::template SingleTreeTraverser<RuleType>
        traverser(rules);

      // Now have it traverse for each point.
      for (size_t i = 0; i < querySet.n_cols; ++i)
        traverser.Traverse(i, *referenceTree);

      numPrunes = traverser.NumPrunes();

      Log::Info << "Single-tree traversal complete." << std::endl;
      Log::Info << "Average number of distance calculations per query point: "
          << (rules.NumDistComputations() / querySet.n_cols) << "."
          << std::endl;
    }
  }
  else // Dual-tree recursion.
  {
    Log::Info << "Performing dual-tree traversal..." << std::endl;

    typedef RASearchRules<SortPolicy, MetricType, TreeType> RuleType;
    RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr,
                   metric, tau, alpha, sampleAtLeaves, firstLeafExact,
                   singleSampleLimit);

    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    if (queryTree)
    {
      Log::Info << "Query statistic pre-search: "
          << queryTree->Stat().NumSamplesMade() << std::endl;
      traverser.Traverse(*queryTree, *referenceTree);
    }
    else
    {
      Log::Info << "Query statistic pre-search: "
          << referenceTree->Stat().NumSamplesMade() << std::endl;
      traverser.Traverse(*referenceTree, *referenceTree);
    }

    numPrunes = traverser.NumPrunes();

    Log::Info << "Dual-tree traversal complete." << std::endl;
    Log::Info << "Average number of distance calculations per query point: "
        << (rules.NumDistComputations() / querySet.n_cols) << "." << std::endl;
  }

  Timer::Stop("computing_neighbors");
  Log::Info << "Pruned " << numPrunes << " nodes." << std::endl;

  // Now, do we need to do mapping of indices?
  if ((!ownReferenceTree && !ownQueryTree) || naive)
  {
    // No mapping needed if we do not own the trees or if we are doing naive
    // sampling.  We are done.
    return;
  }
  else if (ownReferenceTree && ownQueryTree) // Map references and queries.
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
  else if (ownReferenceTree)
  {
    if (!queryTree) // No query tree -- map both references and queries.
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
    else // Map only references.
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
    }

    // Finished with temporary matrix.
    delete neighborPtr;
  }
  else if (ownQueryTree)
  {
    // Set size of matrices correctly.
    resultingNeighbors.set_size(k, querySet.n_cols);
    distances.set_size(k, querySet.n_cols);

    for (size_t i = 0; i < distances.n_cols; i++)
    {
      // Map distances (copy a column).
      distances.col(oldFromNewQueries[i]) = distancePtr->col(i);

      // Map indices of neighbors.
      resultingNeighbors.col(oldFromNewQueries[i]) = neighborPtr->col(i);
    }

    // Finished with temporary matrices.
    delete neighborPtr;
    delete distancePtr;
  }
} // Search

template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearch<SortPolicy, MetricType, TreeType>::ResetQueryTree()
{
  if (!singleMode)
  {
    if (queryTree)
      ResetRAQueryStat(queryTree);
    else
      ResetRAQueryStat(referenceTree);
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearch<SortPolicy, MetricType, TreeType>::ResetRAQueryStat(
    TreeType* treeNode)
{
  treeNode->Stat().Bound() = SortPolicy::WorstDistance();
  treeNode->Stat().NumSamplesMade() = 0;

  for (size_t i = 0; i < treeNode->NumChildren(); i++)
    ResetRAQueryStat(&treeNode->Child(i));
}

}; // namespace neighbor
}; // namespace mlpack

#endif
