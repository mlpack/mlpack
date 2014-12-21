/**
 * @file ra_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of RASearch class to perform rank-approximate
 * all-nearest-neighbors on two specified data sets.
 *
 * This file is part of MLPACK 1.0.9.
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

namespace aux {

//! Call the tree constructor that does mapping.
template<typename TreeType>
TreeType* BuildTree(
    typename TreeType::Mat& dataset,
    std::vector<size_t>& oldFromNew,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  return new TreeType(dataset, oldFromNew);
}

//! Call the tree constructor that does not do mapping.
template<typename TreeType>
TreeType* BuildTree(
    const typename TreeType::Mat& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == false, TreeType*
    >::type = 0)
{
  return new TreeType(dataset);
}

}; // namespace aux

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(const typename TreeType::Mat& referenceSetIn,
         const typename TreeType::Mat& querySetIn,
         const bool naive,
         const bool singleMode,
         const MetricType metric) :
    referenceSet(tree::TreeTraits<TreeType>::RearrangesDataset ? referenceCopy :
        referenceSetIn),
    querySet((tree::TreeTraits<TreeType>::RearrangesDataset && !singleMode) ?
        queryCopy : querySetIn),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(!naive),
    hasQuerySet(true),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // We'll time tree building.
  Timer::Start("tree_building");

  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    referenceCopy = referenceSetIn;
    if (!singleMode)
      queryCopy = querySetIn;
  }

  // Construct as a naive object if we need to.
  if (!naive)
  {
    referenceTree = aux::BuildTree<TreeType>(const_cast<typename
        TreeType::Mat&>(referenceSet), oldFromNewReferences);

    if (!singleMode)
      queryTree = aux::BuildTree<TreeType>(const_cast<typename
          TreeType::Mat&>(querySet), oldFromNewQueries);
  }

  // Stop the timer we started above.
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(const typename TreeType::Mat& referenceSetIn,
         const bool naive,
         const bool singleMode,
         const MetricType metric) :
    referenceSet(tree::TreeTraits<TreeType>::RearrangesDataset ? referenceCopy :
        referenceSetIn),
    querySet(tree::TreeTraits<TreeType>::RearrangesDataset && !singleMode ?
        referenceCopy : referenceSetIn),
    referenceTree(NULL),
    queryTree(NULL),
    treeOwner(!naive),
    hasQuerySet(false),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    numberOfPrunes(0)
{
  // We'll time tree building.
  Timer::Start("tree_building");

  if (tree::TreeTraits<TreeType>::RearrangesDataset)
    referenceCopy = referenceSetIn;

  // Construct as a naive object if we need to.
  if (!naive)
    referenceTree = aux::BuildTree<TreeType>(const_cast<typename
        TreeType::Mat&>(referenceSet), oldFromNewReferences);

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
    treeOwner(false),
    hasQuerySet(true),
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
    treeOwner(false),
    hasQuerySet(false),
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
  if (treeOwner)
  {
    if (referenceTree)
      delete referenceTree;
   if (queryTree)
      delete queryTree;
  }
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

  // Mapping is only required if this tree type rearranges points and we are not
  // in naive mode.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    if (treeOwner && !(singleMode && hasQuerySet))
      distancePtr = new arma::mat; // Query indices need to be mapped.
    if (treeOwner)
      neighborPtr = new arma::Mat<size_t>; // All indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  size_t numPrunes = 0;

  if (naive)
  {
    // We don't need to run the base case on every possible combination of
    // points; we can achieve the rank approximation guarantee with probability
    // alpha by sampling the reference set.
    typedef RASearchRules<SortPolicy, MetricType, TreeType> RuleType;
    RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr,
                   metric, tau, alpha, naive, sampleAtLeaves, firstLeafExact,
                   singleSampleLimit);

    // Find how many samples from the reference set we need and sample uniformly
    // from the reference set without replacement.
    const size_t numSamples = rules.MinimumSamplesReqd(referenceSet.n_cols, k,
        tau, alpha);
    arma::uvec distinctSamples;
    rules.ObtainDistinctSamples(numSamples, referenceSet.n_cols,
        distinctSamples);

    // Run the base case on each combination of query point and sampled
    // reference point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < distinctSamples.n_elem; ++j)
        rules.BaseCase(i, (size_t) distinctSamples[j]);
  }
  else if (singleMode)
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
  if (!treeOwner || !tree::TreeTraits<TreeType>::RearrangesDataset)
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
    // No query tree -- map both references and queries.
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

// Returns a String of the Object.
template<typename SortPolicy, typename MetricType, typename TreeType>
std::string RASearch<SortPolicy, MetricType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "RA Search  [" << this << "]" << std::endl;
  convert << "  Reference Set: " << referenceSet.n_rows << "x" ;
  convert <<  referenceSet.n_cols << std::endl;
  if (&referenceSet != &querySet)
    convert << "  QuerySet: " << querySet.n_rows << "x" << querySet.n_cols
        << std::endl;
  if (naive)
    convert << "  Naive: TRUE" << std::endl;
  if (singleMode)
    convert << "  Single Node: TRUE" << std::endl;
  convert << "  Metric: " << std::endl <<
      mlpack::util::Indent(metric.ToString(),2);
  return convert.str();
}

}; // namespace neighbor
}; // namespace mlpack

#endif
