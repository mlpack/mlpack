/**
 * @file ra_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of RASearch class to perform rank-approximate
 * all-nearest-neighbors on two specified data sets.
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

} // namespace aux

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(const typename TreeType::Mat& referenceSetIn,
         const bool naive,
         const bool singleMode,
         const double tau,
         const double alpha,
         const bool sampleAtLeaves,
         const bool firstLeafExact,
         const size_t singleSampleLimit,
         const MetricType metric) :
    referenceSet((tree::TreeTraits<TreeType>::RearrangesDataset && !naive)
        ? referenceCopy : referenceSetIn),
    referenceTree(NULL),
    treeOwner(!naive),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    tau(tau),
    alpha(alpha),
    sampleAtLeaves(sampleAtLeaves),
    firstLeafExact(firstLeafExact),
    singleSampleLimit(singleSampleLimit),
    metric(metric)
{
  // We'll time tree building.
  Timer::Start("tree_building");

  if (!naive)
  {
    if (tree::TreeTraits<TreeType>::RearrangesDataset)
      referenceCopy = referenceSetIn;

    referenceTree = aux::BuildTree<TreeType>(
        const_cast<typename TreeType::Mat&>(referenceSet),
        oldFromNewReferences);
  }

  // Stop the timer we started above.
  Timer::Stop("tree_building");
}

// Construct the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
RASearch(TreeType* referenceTree,
         const bool singleMode,
         const double tau,
         const double alpha,
         const bool sampleAtLeaves,
         const bool firstLeafExact,
         const size_t singleSampleLimit,
         const MetricType metric) :
    referenceSet(referenceTree->Dataset()),
    referenceTree(referenceTree),
    treeOwner(false),
    naive(false),
    singleMode(singleMode),
    tau(tau),
    alpha(alpha),
    sampleAtLeaves(sampleAtLeaves),
    firstLeafExact(firstLeafExact),
    singleSampleLimit(singleSampleLimit),
    metric(metric)
// Nothing else to initialize.
{  }

/**
 * The tree is the only member we may be responsible for deleting.  The others
 * will take care of themselves.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
RASearch<SortPolicy, MetricType, TreeType>::
~RASearch()
{
  if (treeOwner && referenceTree)
    delete referenceTree;
}

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearch<SortPolicy, MetricType, TreeType>::
Search(const typename TreeType::Mat& querySet,
       const size_t k,
       arma::Mat<size_t>& neighbors,
       arma::mat& distances)
{
  Timer::Start("computing_neighbors");

  // This will hold mappings for query points, if necessary.
  std::vector<size_t> oldFromNewQueries;

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid an extra copy, we will store the neighbors and distances in a
  // separate matrix.
  arma::Mat<size_t>* neighborPtr = &neighbors;
  arma::mat* distancePtr = &distances;

  // Mapping is only required if this tree type rearranges points and we are not
  // in naive mode.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    if (!singleMode && !naive)
      distancePtr = new arma::mat; // Query indices need to be mapped.

    if (treeOwner)
      neighborPtr = new arma::Mat<size_t>; // All indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  // If we will be building a tree and it will modify the query set, make a copy
  // of the dataset.
  typename TreeType::Mat queryCopy;
  const bool needsCopy = (!naive && !singleMode &&
      tree::TreeTraits<TreeType>::RearrangesDataset);
  if (needsCopy)
    queryCopy = querySet;

  const typename TreeType::Mat& querySetRef = (needsCopy) ? queryCopy :
      querySet;

  // Create the helper object for the tree traversal.
  typedef RASearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, querySetRef, *neighborPtr, *distancePtr,
                 metric, tau, alpha, naive, sampleAtLeaves, firstLeafExact,
                 singleSampleLimit, false);

  if (naive)
  {
    // Find how many samples from the reference set we need and sample uniformly
    // from the reference set without replacement.
    const size_t numSamples = rules.MinimumSamplesReqd(referenceSet.n_cols, k,
        tau, alpha);
    arma::uvec distinctSamples;
    rules.ObtainDistinctSamples(numSamples, referenceSet.n_cols,
        distinctSamples);

    // Run the base case on each combination of query point and sampled
    // reference point.
    for (size_t i = 0; i < querySetRef.n_cols; ++i)
      for (size_t j = 0; j < distinctSamples.n_elem; ++j)
        rules.BaseCase(i, (size_t) distinctSamples[j]);
  }
  else if (singleMode)
  {
    // If the reference root node is a leaf, then the sampling has already been
    // done in the RASearchRules constructor.  This happens when naive = true.
    if (!referenceTree->IsLeaf())
    {
      Log::Info << "Performing single-tree traversal..." << std::endl;

      // Create the traverser.
      typename TreeType::template SingleTreeTraverser<RuleType>
        traverser(rules);

      // Now have it traverse for each point.
      for (size_t i = 0; i < querySetRef.n_cols; ++i)
        traverser.Traverse(i, *referenceTree);

      Log::Info << "Single-tree traversal complete." << std::endl;
      Log::Info << "Average number of distance calculations per query point: "
          << (rules.NumDistComputations() / querySet.n_cols) << "."
          << std::endl;
    }
  }
  else // Dual-tree recursion.
  {
    Log::Info << "Performing dual-tree traversal..." << std::endl;

    // Build the query tree.
    Timer::Stop("computing_neighbors");
    Timer::Start("tree_building");
    TreeType* queryTree = aux::BuildTree<TreeType>(
        const_cast<typename TreeType::Mat&>(querySetRef), oldFromNewQueries);
    Timer::Stop("tree_building");
    Timer::Start("computing_neighbors");

    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    Log::Info << "Query statistic pre-search: "
        << queryTree->Stat().NumSamplesMade() << std::endl;

    traverser.Traverse(*queryTree, *referenceTree);

    Log::Info << "Dual-tree traversal complete." << std::endl;
    Log::Info << "Average number of distance calculations per query point: "
        << (rules.NumDistComputations() / querySet.n_cols) << "." << std::endl;
  }

  Timer::Stop("computing_neighbors");

  // Map points back to original indices, if necessary.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    if (!singleMode && !naive && treeOwner)
    {
      // We must map both query and reference indices.
      neighbors.set_size(k, querySet.n_cols);
      distances.set_size(k, querySet.n_cols);

      for (size_t i = 0; i < distances.n_cols; i++)
      {
        // Map distances (copy a column).
        distances.col(oldFromNewQueries[i]) = distancePtr->col(i);

        // Map indices of neighbors.
        for (size_t j = 0; j < distances.n_rows; j++)
        {
          neighbors(j, oldFromNewQueries[i]) =
              oldFromNewReferences[(*neighborPtr)(j, i)];
        }
      }

      // Finished with temporary matrices.
      delete neighborPtr;
      delete distancePtr;
    }
    else if (!singleMode && !naive)
    {
      // We must map query indices only.
      neighbors.set_size(k, querySet.n_cols);
      distances.set_size(k, querySet.n_cols);

      for (size_t i = 0; i < distances.n_cols; ++i)
      {
        // Map distances (copy a column).
        const size_t queryMapping = oldFromNewQueries[i];
        distances.col(queryMapping) = distancePtr->col(i);
        neighbors.col(queryMapping) = neighborPtr->col(i);
      }

      // Finished with temporary matrices.
      delete neighborPtr;
      delete distancePtr;
    }
    else if (treeOwner)
    {
      // We must map reference indices only.
      neighbors.set_size(k, querySet.n_cols);

      // Map indices of neighbors.
      for (size_t i = 0; i < neighbors.n_cols; i++)
        for (size_t j = 0; j < neighbors.n_rows; j++)
          neighbors(j, i) = oldFromNewReferences[(*neighborPtr)(j, i)];

      // Finished with temporary matrix.
      delete neighborPtr;
    }
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearch<SortPolicy, MetricType, TreeType>::Search(
    TreeType* queryTree,
    const size_t k,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  Timer::Start("computing_neighbors");

  // Get a reference to the query set.
  const typename TreeType::Mat& querySet = queryTree->Dataset();

  // Make sure we are in dual-tree mode.
  if (singleMode || naive)
    throw std::invalid_argument("cannot call NeighborSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  // We won't need to map query indices, but will we need to map distances?
  arma::Mat<size_t>* neighborPtr = &neighbors;

  if (treeOwner && tree::TreeTraits<TreeType>::RearrangesDataset)
    neighborPtr = new arma::Mat<size_t>;

  neighborPtr->set_size(k, querySet.n_cols);
  neighborPtr->fill(size_t() - 1);
  distances.set_size(k, querySet.n_cols);
  distances.fill(SortPolicy::WorstDistance());

  // Create the helper object for the tree traversal.
  typedef RASearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, queryTree->Dataset(), *neighborPtr, distances,
                 metric, tau, alpha, naive, sampleAtLeaves, firstLeafExact,
                 singleSampleLimit, false);

  // Create the traverser.
  typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);
  traverser.Traverse(*queryTree, *referenceTree);

  Timer::Stop("computing_neighbors");

  // Do we need to map indices?
  if (treeOwner && tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    // We must map reference indices only.
    neighbors.set_size(k, querySet.n_cols);

    // Map indices of neighbors.
    for (size_t i = 0; i < neighbors.n_cols; i++)
      for (size_t j = 0; j < neighbors.n_rows; j++)
        neighbors(j, i) = oldFromNewReferences[(*neighborPtr)(j, i)];

    // Finished with temporary matrix.
    delete neighborPtr;
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearch<SortPolicy, MetricType, TreeType>::Search(
    const size_t k,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  Timer::Start("computing_neighbors");

  arma::Mat<size_t>* neighborPtr = &neighbors;
  arma::mat* distancePtr = &distances;

  if (tree::TreeTraits<TreeType>::RearrangesDataset && treeOwner)
  {
    // We will always need to rearrange in this case.
    distancePtr = new arma::mat;
    neighborPtr = new arma::Mat<size_t>;
  }

  // Initialize results.
  neighborPtr->set_size(k, referenceSet.n_cols);
  neighborPtr->fill(size_t() - 1);
  distancePtr->set_size(k, referenceSet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  // Create the helper object for the tree traversal.
  typedef RASearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, referenceSet, *neighborPtr, *distancePtr,
                 metric, tau, alpha, naive, sampleAtLeaves, firstLeafExact,
                 singleSampleLimit, true /* sets are the same */);

  if (naive)
  {
    // Find how many samples from the reference set we need and sample uniformly
    // from the reference set without replacement.
    const size_t numSamples = rules.MinimumSamplesReqd(referenceSet.n_cols, k,
        tau, alpha);
    arma::uvec distinctSamples;
    rules.ObtainDistinctSamples(numSamples, referenceSet.n_cols,
        distinctSamples);

    // The naive brute-force solution.
    for (size_t i = 0; i < referenceSet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet.n_cols; ++j)
        rules.BaseCase(i, j);
  }
  else if (singleMode)
  {
    // Create the traverser.
    typename TreeType::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < referenceSet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);
  }
  else
  {
    // Create the traverser.
    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*referenceTree, *referenceTree);
  }

  Timer::Stop("computing_neighbors");

  // Do we need to map the reference indices?
  if (treeOwner && tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    neighbors.set_size(k, referenceSet.n_cols);
    distances.set_size(k, referenceSet.n_cols);

    for (size_t i = 0; i < distances.n_cols; ++i)
    {
      // Map distances (copy a column).
      const size_t refMapping = oldFromNewReferences[i];
      distances.col(refMapping) = distancePtr->col(i);

      // Map each neighbor's index.
      for (size_t j = 0; j < distances.n_rows; ++j)
        neighbors(j, refMapping) = oldFromNewReferences[(*neighborPtr)(j, i)];
    }

    // Finished with temporary matrices.
    delete neighborPtr;
    delete distancePtr;
  }
}

template<typename SortPolicy, typename MetricType, typename TreeType>
void RASearch<SortPolicy, MetricType, TreeType>::ResetQueryTree(
    TreeType* queryNode) const
{
  queryNode->Stat().Bound() = SortPolicy::WorstDistance();
  queryNode->Stat().NumSamplesMade() = 0;

  for (size_t i = 0; i < queryNode->NumChildren(); i++)
    ResetQueryTree(&queryNode->Child(i));
}

// Returns a string representation of the object.
template<typename SortPolicy, typename MetricType, typename TreeType>
std::string RASearch<SortPolicy, MetricType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "RASearch [" << this << "]" << std::endl;
  convert << "  referenceSet: " << referenceSet.n_rows << "x"
      << referenceSet.n_cols << std::endl;

  convert << "  naive: ";
  if (naive)
    convert << "true" << std::endl;
  else
    convert << "false" << std::endl;

  convert << "  singleMode: ";
  if (singleMode)
    convert << "true" << std::endl;
  else
    convert << "false" << std::endl;

  convert << "  tau: " << tau << std::endl;
  convert << "  alpha: " << alpha << std::endl;
  convert << "  sampleAtLeaves: ";
  if (sampleAtLeaves)
    convert << "true" << std::endl;
  else
    convert << "false" << std::endl;

  convert << "  firstLeafExact: ";
  if (firstLeafExact)
    convert << "true" << std::endl;
  else
    convert << "false" << std::endl;
  convert << "  singleSampleLimit: " << singleSampleLimit << std::endl;
  convert << "  metric: " << std::endl <<
      mlpack::util::Indent(metric.ToString(),2);
  return convert.str();
}

}; // namespace neighbor
}; // namespace mlpack

#endif
