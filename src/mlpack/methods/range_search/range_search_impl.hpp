/**
 * @file range_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the RangeSearch class.
 */
#ifndef __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_IMPL_HPP
#define __MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_IMPL_HPP

// Just in case it hasn't been included.
#include "range_search.hpp"

// The rules for traversal.
#include "range_search_rules.hpp"

namespace mlpack {
namespace range {

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    const typename TreeType::Mat& referenceSet,
    const typename TreeType::Mat& querySet,
    const bool naive,
    const bool singleMode,
    const size_t leafSize,
    const MetricType metric) :
    referenceCopy(referenceSet),
    queryCopy(querySet),
    referenceSet(referenceCopy),
    querySet(queryCopy),
    ownReferenceTree(true),
    ownQueryTree(true),
    naive(naive),
    singleMode(!naive && singleMode), // Naive overrides single mode.
    metric(metric),
    numPrunes(0)
{
  // Build the trees.
  Timer::Start("range_search/tree_building");

  // Naive sets the leaf size such that the entire tree is one node.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceCopy.n_cols : leafSize));

  queryTree = new TreeType(queryCopy, oldFromNewQueries,
      (naive ? queryCopy.n_cols : leafSize));

  Timer::Stop("range_search/tree_building");
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    const typename TreeType::Mat& referenceSet,
    const bool naive,
    const bool singleMode,
    const size_t leafSize,
    const MetricType metric) :
    referenceCopy(referenceSet),
    referenceSet(referenceCopy),
    querySet(referenceCopy),
    queryTree(NULL),
    ownReferenceTree(true),
    ownQueryTree(false),
    naive(naive),
    singleMode(!naive && singleMode), // Naive overrides single mode.
    metric(metric),
    numPrunes(0)
{
  // Build the trees.
  Timer::Start("range_search/tree_building");

  // Naive sets the leaf size such that the entire tree is one node.
  referenceTree = new TreeType(referenceCopy, oldFromNewReferences,
      (naive ? referenceCopy.n_cols : leafSize));

  Timer::Stop("range_search/tree_building");
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
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
    ownReferenceTree(false),
    ownQueryTree(false),
    naive(false),
    singleMode(singleMode),
    metric(metric),
    numPrunes(0)
{
  // Nothing else to initialize.
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    TreeType* referenceTree,
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
    numPrunes(0)
{
  // Nothing else to initialize.
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::~RangeSearch()
{
  if (ownReferenceTree)
    delete referenceTree;
  if (ownQueryTree)
    delete queryTree;
}

template<typename MetricType, typename TreeType>
void RangeSearch<MetricType, TreeType>::Search(
    const math::Range& range,
    std::vector<std::vector<size_t> >& neighbors,
    std::vector<std::vector<double> >& distances)
{
  Timer::Start("range_search/computing_neighbors");

  // Set size of prunes to 0.
  numPrunes = 0;

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid extra copies, we will store the unmapped neighbors and distances
  // in a separate object.
  std::vector<std::vector<size_t> >* neighborPtr = &neighbors;
  std::vector<std::vector<double> >* distancePtr = &distances;

  if (ownQueryTree || (ownReferenceTree && !queryTree))
    distancePtr = new std::vector<std::vector<double> >;
  if (ownReferenceTree || ownQueryTree)
    neighborPtr = new std::vector<std::vector<size_t> >;

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distancePtr->clear();
  distancePtr->resize(querySet.n_cols);

  // Create the helper object for the traversal.
  typedef RangeSearchRules<MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, querySet, range, *neighborPtr, *distancePtr,
      metric);

  if (singleMode)
  {
    // Create the traverser.
    typename TreeType::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    numPrunes = traverser.NumPrunes();
  }
  else // Dual-tree recursion.
  {
    // Create the traverser.
    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    if (queryTree)
      traverser.Traverse(*queryTree, *referenceTree);
    else
      traverser.Traverse(*referenceTree, *referenceTree);

    numPrunes = traverser.NumPrunes();
  }

  Timer::Stop("range_search/computing_neighbors");

  // Output number of prunes.
  Log::Info << "Number of pruned nodes during computation: " << numPrunes
      << "." << std::endl;

  // Map points back to original indices, if necessary.
  if (!ownReferenceTree && !ownQueryTree)
  {
    // No mapping needed.  We are done.
    return;
  }
  else if (ownReferenceTree && ownQueryTree) // Map references and queries.
  {
    neighbors.clear();
    neighbors.resize(querySet.n_cols);
    distances.clear();
    distances.resize(querySet.n_cols);

    for (size_t i = 0; i < distances.size(); i++)
    {
      // Map distances (copy a column).
      size_t queryMapping = oldFromNewQueries[i];
      distances[queryMapping] = (*distancePtr)[i];

      // Copy each neighbor individually, because we need to map it.
      neighbors[queryMapping].resize(distances[queryMapping].size());
      for (size_t j = 0; j < distances[queryMapping].size(); j++)
      {
        neighbors[queryMapping][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
      }
    }

    // Finished with temporary objects.
    delete neighborPtr;
    delete distancePtr;
  }
  else if (ownReferenceTree)
  {
    if (!queryTree) // No query tree -- map both references and queries.
    {
      neighbors.clear();
      neighbors.resize(querySet.n_cols);
      distances.clear();
      distances.resize(querySet.n_cols);

      for (size_t i = 0; i < distances.size(); i++)
      {
        // Map distances (copy a column).
        size_t refMapping = oldFromNewReferences[i];
        distances[refMapping] = (*distancePtr)[i];

        // Copy each neighbor individually, because we need to map it.
        neighbors[refMapping].resize(distances[refMapping].size());
        for (size_t j = 0; j < distances[refMapping].size(); j++)
        {
          neighbors[refMapping][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
        }
      }

      // Finished with temporary objects.
      delete neighborPtr;
      delete distancePtr;
    }
    else // Map only references.
    {
      neighbors.clear();
      neighbors.resize(querySet.n_cols);

      // Map indices of neighbors.
      for (size_t i = 0; i < neighbors.size(); i++)
      {
        neighbors[i].resize((*neighborPtr)[i].size());
        for (size_t j = 0; j < neighbors[i].size(); j++)
        {
          neighbors[i][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
        }
      }

      // Finished with temporary object.
      delete neighborPtr;
    }
  }
  else if (ownQueryTree)
  {
    neighbors.clear();
    neighbors.resize(querySet.n_cols);
    distances.clear();
    distances.resize(querySet.n_cols);

    for (size_t i = 0; i < distances.size(); i++)
    {
      // Map distances (copy a column).
      distances[oldFromNewQueries[i]] = (*distancePtr)[i];

      // Map neighbors.
      neighbors[oldFromNewQueries[i]] = (*neighborPtr)[i];
    }

    // Finished with temporary objects.
    delete neighborPtr;
    delete distancePtr;
  }
}

}; // namespace range
}; // namespace mlpack

#endif
