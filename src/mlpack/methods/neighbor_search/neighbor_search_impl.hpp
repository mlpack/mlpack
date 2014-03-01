/**
 * @file neighbor_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Neighbor-Search class to perform all-nearest-neighbors on
 * two specified data sets.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

#include "neighbor_search_rules.hpp"

namespace mlpack {
namespace neighbor {

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
    treeOwner(!naive), // False if a tree was passed.  If naive, then no trees.
    hasQuerySet(true),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric)
{
  // C++11 will allow us to call out to other constructors so we can avoid this
  // copypasta problem.

  // We'll time tree building, but only if we are building trees.
  Timer::Start("tree_building");

  // If not in naive mode, then we need to build trees.
  if (!naive)
  {
    referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
        (naive ? referenceCopy.n_cols : leafSize));

    if (!singleMode)
      queryTree = new TreeType(queryCopy, oldFromNewQueries,
          (naive ? querySet.n_cols : leafSize));
  }

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
    treeOwner(!naive), // If naive, then we are not building any trees.
    hasQuerySet(false),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric)
{
  // We'll time tree building, but only if we are building trees.
  Timer::Start("tree_building");

  // If not in naive mode, then we may need to construct trees.
  if (!naive)
  {
    referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
        (naive ? referenceSet.n_cols : leafSize));
    if (!singleMode)
      queryTree = new TreeType(*referenceTree);
  }

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
    metric(metric)
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
    metric(metric)
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
  else if (!treeOwner && !hasQuerySet && !(singleMode || naive))
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
  neighborPtr->fill(size_t() - 1);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  // Create the helper object for the tree traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr, metric);

  if (naive)
  {
    // The naive brute-force traversal.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet.n_cols; ++j)
        rules.BaseCase(i, j);
  }
  else if (singleMode)
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

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";
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

    // Finished with temporary matrices.
    delete neighborPtr;
    delete distancePtr;
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


//Return a String of the Object.
template<typename SortPolicy, typename MetricType, typename TreeType>
std::string NeighborSearch<SortPolicy, MetricType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "NearestNeighborSearch [" << this << "]" << std::endl;
  convert << "  Reference Set: " << referenceSet.n_rows << "x" ;
  convert <<  referenceSet.n_cols << std::endl;
  if (&referenceSet != &querySet)
    convert << "  QuerySet: " << querySet.n_rows << "x" << querySet.n_cols
        << std::endl;
  convert << "  Reference Tree: " << referenceTree << std::endl;
  if (&referenceTree != &queryTree)
    convert << "  QueryTree: " << queryTree << std::endl;
  convert << "  Tree Owner: " << treeOwner << std::endl;
  convert << "  Naive: " << naive << std::endl;
  convert << "  Metric: " << std::endl;
  convert << mlpack::util::Indent(metric.ToString(),2);
  return convert.str();
}

}; // namespace neighbor
}; // namespace mlpack

#endif
