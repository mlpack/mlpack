/**
 * @file ra_search_impl.hpp
 * @author Parikshit Ram
 *
 * Implementation of RASearch class to perform rank-approximate
 * all-nearest-neighbors on two specified data sets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_SEARCH_IMPL_HPP
#define MLPACK_METHODS_RANN_RA_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

#include "ra_search_rules.hpp"

namespace mlpack {
namespace neighbor {

namespace aux {

//! Call the tree constructor that does mapping.
template<typename TreeType>
TreeType* BuildTree(
    const typename TreeType::Mat& dataset,
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

//! Call the tree constructor that does mapping.
template<typename TreeType>
TreeType* BuildTree(
    typename TreeType::Mat&& dataset,
    std::vector<size_t>& oldFromNew,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  return new TreeType(std::move(dataset), oldFromNew);
}

//! Call the tree constructor that does not do mapping.
template<typename TreeType>
TreeType* BuildTree(
    typename TreeType::Mat&& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == false, TreeType*
    >::type = 0)
{
  return new TreeType(std::move(dataset));
}

} // namespace aux

// Construct the object.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RASearch<SortPolicy, MetricType, MatType, TreeType>::
RASearch(const MatType& referenceSetIn,
         const bool naive,
         const bool singleMode,
         const double tau,
         const double alpha,
         const bool sampleAtLeaves,
         const bool firstLeafExact,
         const size_t singleSampleLimit,
         const MetricType metric) :
    referenceTree(naive ? NULL : aux::BuildTree<Tree>(
        const_cast<MatType&>(referenceSetIn), oldFromNewReferences)),
    referenceSet(naive ? &referenceSetIn : &referenceTree->Dataset()),
    treeOwner(!naive),
    setOwner(false),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    tau(tau),
    alpha(alpha),
    sampleAtLeaves(sampleAtLeaves),
    firstLeafExact(firstLeafExact),
    singleSampleLimit(singleSampleLimit),
    metric(metric)
{
  // Nothing to do.
}

// Construct the object, taking ownership of the data matrix.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RASearch<SortPolicy, MetricType, MatType, TreeType>::
RASearch(MatType&& referenceSetIn,
         const bool naive,
         const bool singleMode,
         const double tau,
         const double alpha,
         const bool sampleAtLeaves,
         const bool firstLeafExact,
         const size_t singleSampleLimit,
         const MetricType metric) :
    referenceTree(naive ? NULL : aux::BuildTree<Tree>(
        std::move(referenceSetIn), oldFromNewReferences)),
    referenceSet(naive ? new MatType(std::move(referenceSetIn)) :
        &referenceTree->Dataset()),
    treeOwner(!naive),
    setOwner(naive),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    tau(tau),
    alpha(alpha),
    sampleAtLeaves(sampleAtLeaves),
    firstLeafExact(firstLeafExact),
    singleSampleLimit(singleSampleLimit),
    metric(metric)
{
  // Nothing to do.
}

// Construct the object.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RASearch<SortPolicy, MetricType, MatType, TreeType>::
RASearch(Tree* referenceTree,
         const bool singleMode,
         const double tau,
         const double alpha,
         const bool sampleAtLeaves,
         const bool firstLeafExact,
         const size_t singleSampleLimit,
         const MetricType metric) :
    referenceTree(referenceTree),
    referenceSet(&referenceTree->Dataset()),
    treeOwner(false),
    setOwner(false),
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

// Empty constructor.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RASearch<SortPolicy, MetricType, MatType, TreeType>::
RASearch(const bool naive,
         const bool singleMode,
         const double tau,
         const double alpha,
         const bool sampleAtLeaves,
         const bool firstLeafExact,
         const size_t singleSampleLimit,
         const MetricType metric) :
    referenceTree(NULL),
    referenceSet(new MatType()),
    treeOwner(false),
    setOwner(true),
    naive(naive),
    singleMode(singleMode),
    tau(tau),
    alpha(alpha),
    sampleAtLeaves(sampleAtLeaves),
    firstLeafExact(firstLeafExact),
    singleSampleLimit(singleSampleLimit),
    metric(metric)
{
  // Build the tree on the empty dataset, if necessary.
  if (!naive)
  {
    referenceTree = aux::BuildTree<Tree>(*referenceSet, oldFromNewReferences);
    treeOwner = true;
  }
}

/**
 * The tree and the dataset are the only members we may be responsible for
 * deleting.  The others will take care of themselves.
 */
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RASearch<SortPolicy, MetricType, MatType, TreeType>::
~RASearch()
{
  if (treeOwner && referenceTree)
    delete referenceTree;
  if (setOwner)
    delete referenceSet;
}

// Train on a new reference set.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RASearch<SortPolicy, MetricType, MatType, TreeType>::Train(
    const MatType& referenceSet)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
    delete referenceTree;

  // We may need to rebuild the tree.
  if (!naive)
  {
    referenceTree = aux::BuildTree<Tree>(referenceSet, oldFromNewReferences);
    treeOwner = true;
  }
  else
  {
    treeOwner = false;
  }

  // Delete the old reference set, if we owned it.
  if (setOwner && this->referenceSet)
    delete this->referenceSet;

  if (!naive)
    this->referenceSet = &referenceTree->Dataset();
  else
    this->referenceSet = &referenceSet;
  setOwner = false; // We don't own the set in either case.
}

// Train on a new reference set.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RASearch<SortPolicy, MetricType, MatType, TreeType>::Train(
    MatType&& referenceSet)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
    delete referenceTree;

  // We may need to rebuild the tree.
  if (!naive)
  {
    referenceTree = aux::BuildTree<Tree>(std::move(referenceSet),
        oldFromNewReferences);
    treeOwner = true;
  }
  else
  {
    treeOwner = false;
  }

  // Delete the old reference set, if we owned it.
  if (setOwner && this->referenceSet)
    delete this->referenceSet;

  if (!naive)
  {
    this->referenceSet = &referenceTree->Dataset();
    setOwner = false;
  }
  else
  {
    this->referenceSet = new MatType(std::move(referenceSet));
    setOwner = true;
  }
}

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RASearch<SortPolicy, MetricType, MatType, TreeType>::
Search(const MatType& querySet,
       const size_t k,
       arma::Mat<size_t>& neighbors,
       arma::mat& distances)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }

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
  if (tree::TreeTraits<Tree>::RearrangesDataset)
  {
    if (!singleMode && !naive)
    {
      distancePtr = new arma::mat; // Query indices need to be mapped.
      neighborPtr = new arma::Mat<size_t>;
    }

    else if (treeOwner)
      neighborPtr = new arma::Mat<size_t>; // All indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);

  typedef RASearchRules<SortPolicy, MetricType, Tree> RuleType;

  if (naive)
  {
    RuleType rules(*referenceSet, querySet, k, metric, tau, alpha, naive,
        sampleAtLeaves, firstLeafExact, singleSampleLimit, false);

    // Find how many samples from the reference set we need and sample uniformly
    // from the reference set without replacement.
    const size_t numSamples = RAUtil::MinimumSamplesReqd(referenceSet->n_cols,
        k, tau, alpha);
    arma::uvec distinctSamples;
    math::ObtainDistinctSamples(0, referenceSet->n_cols, numSamples,
        distinctSamples);

    // Run the base case on each combination of query point and sampled
    // reference point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < distinctSamples.n_elem; ++j)
        rules.BaseCase(i, (size_t) distinctSamples[j]);

    rules.GetResults(*neighborPtr, *distancePtr);
  }
  else if (singleMode)
  {
    RuleType rules(*referenceSet, querySet, k, metric, tau, alpha, naive,
        sampleAtLeaves, firstLeafExact, singleSampleLimit, false);

    // If the reference root node is a leaf, then the sampling has already been
    // done in the RASearchRules constructor.  This happens when naive = true.
    if (!referenceTree->IsLeaf())
    {
      Log::Info << "Performing single-tree traversal..." << std::endl;

      // Create the traverser.
      typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

      // Now have it traverse for each point.
      for (size_t i = 0; i < querySet.n_cols; ++i)
        traverser.Traverse(i, *referenceTree);

      Log::Info << "Single-tree traversal complete." << std::endl;
      Log::Info << "Average number of distance calculations per query point: "
          << (rules.NumDistComputations() / querySet.n_cols) << "."
          << std::endl;
    }

    rules.GetResults(*neighborPtr, *distancePtr);
  }
  else // Dual-tree recursion.
  {
    Log::Info << "Performing dual-tree traversal..." << std::endl;

    // Build the query tree.
    Timer::Stop("computing_neighbors");
    Timer::Start("tree_building");
    Tree* queryTree = aux::BuildTree<Tree>(const_cast<MatType&>(querySet),
        oldFromNewQueries);
    Timer::Stop("tree_building");
    Timer::Start("computing_neighbors");

    RuleType rules(*referenceSet, queryTree->Dataset(), k, metric, tau, alpha,
        naive, sampleAtLeaves, firstLeafExact, singleSampleLimit, false);
    typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

    Log::Info << "Query statistic pre-search: "
        << queryTree->Stat().NumSamplesMade() << std::endl;

    traverser.Traverse(*queryTree, *referenceTree);

    Log::Info << "Dual-tree traversal complete." << std::endl;
    Log::Info << "Average number of distance calculations per query point: "
        << (rules.NumDistComputations() / querySet.n_cols) << "." << std::endl;

    rules.GetResults(*neighborPtr, *distancePtr);

    delete queryTree;
  }

  Timer::Stop("computing_neighbors");

  // Map points back to original indices, if necessary.
  if (tree::TreeTraits<Tree>::RearrangesDataset)
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

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RASearch<SortPolicy, MetricType, MatType, TreeType>::Search(
    Tree* queryTree,
    const size_t k,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  Timer::Start("computing_neighbors");

  // Get a reference to the query set.
  const MatType& querySet = queryTree->Dataset();

  // Make sure we are in dual-tree mode.
  if (singleMode || naive)
    throw std::invalid_argument("cannot call NeighborSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  // We won't need to map query indices, but will we need to map distances?
  arma::Mat<size_t>* neighborPtr = &neighbors;

  if (treeOwner && tree::TreeTraits<Tree>::RearrangesDataset)
    neighborPtr = new arma::Mat<size_t>;

  neighborPtr->set_size(k, querySet.n_cols);
  distances.set_size(k, querySet.n_cols);

  // Create the helper object for the tree traversal.
  typedef RASearchRules<SortPolicy, MetricType, Tree> RuleType;
  RuleType rules(*referenceSet, queryTree->Dataset(), k, metric, tau, alpha,
      naive, sampleAtLeaves, firstLeafExact, singleSampleLimit, false);

  // Create the traverser.
  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);
  traverser.Traverse(*queryTree, *referenceTree);

  rules.GetResults(*neighborPtr, distances);

  Timer::Stop("computing_neighbors");

  // Do we need to map indices?
  if (treeOwner && tree::TreeTraits<Tree>::RearrangesDataset)
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

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RASearch<SortPolicy, MetricType, MatType, TreeType>::Search(
    const size_t k,
    arma::Mat<size_t>& neighbors,
    arma::mat& distances)
{
  Timer::Start("computing_neighbors");

  arma::Mat<size_t>* neighborPtr = &neighbors;
  arma::mat* distancePtr = &distances;

  if (tree::TreeTraits<Tree>::RearrangesDataset && treeOwner)
  {
    // We will always need to rearrange in this case.
    distancePtr = new arma::mat;
    neighborPtr = new arma::Mat<size_t>;
  }

  // Initialize results.
  neighborPtr->set_size(k, referenceSet->n_cols);
  distancePtr->set_size(k, referenceSet->n_cols);

  // Create the helper object for the tree traversal.
  typedef RASearchRules<SortPolicy, MetricType, Tree> RuleType;
  RuleType rules(*referenceSet, *referenceSet, k, metric, tau, alpha, naive,
      sampleAtLeaves, firstLeafExact, singleSampleLimit, true /* same sets */);

  if (naive)
  {
    // Find how many samples from the reference set we need and sample uniformly
    // from the reference set without replacement.
    const size_t numSamples = RAUtil::MinimumSamplesReqd(referenceSet->n_cols,
        k, tau, alpha);
    arma::uvec distinctSamples;
    math::ObtainDistinctSamples(0, referenceSet->n_cols, numSamples,
        distinctSamples);

    // The naive brute-force solution.
    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      for (size_t j = 0; j < referenceSet->n_cols; ++j)
        rules.BaseCase(i, j);
  }
  else if (singleMode)
  {
    // Create the traverser.
    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      traverser.Traverse(i, *referenceTree);
  }
  else
  {
    // Create the traverser.
    typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*referenceTree, *referenceTree);
  }

  rules.GetResults(*neighborPtr, *distancePtr);

  Timer::Stop("computing_neighbors");

  // Do we need to map the reference indices?
  if (treeOwner && tree::TreeTraits<Tree>::RearrangesDataset)
  {
    neighbors.set_size(k, referenceSet->n_cols);
    distances.set_size(k, referenceSet->n_cols);

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

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RASearch<SortPolicy, MetricType, MatType, TreeType>::ResetQueryTree(
    Tree* queryNode) const
{
  queryNode->Stat().Bound() = SortPolicy::WorstDistance();
  queryNode->Stat().NumSamplesMade() = 0;

  for (size_t i = 0; i < queryNode->NumChildren(); i++)
    ResetQueryTree(&queryNode->Child(i));
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
template<typename Archive>
void RASearch<SortPolicy, MetricType, MatType, TreeType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  using data::CreateNVP;

  // Serialize preferences for search.
  ar & CreateNVP(naive, "naive");
  ar & CreateNVP(singleMode, "singleMode");

  ar & CreateNVP(tau, "tau");
  ar & CreateNVP(alpha, "alpha");
  ar & CreateNVP(sampleAtLeaves, "sampleAtLeaves");
  ar & CreateNVP(firstLeafExact, "firstLeafExact");
  ar & CreateNVP(singleSampleLimit, "singleSampleLimit");

  // If we are doing naive search, we serialize the dataset.  Otherwise we
  // serialize the tree.
  if (naive)
  {
    if (Archive::is_loading::value)
    {
      if (setOwner && referenceSet)
        delete referenceSet;

      setOwner = true;
    }

    ar & CreateNVP(referenceSet, "referenceSet");
    ar & CreateNVP(metric, "metric");

    // If we are loading, set the tree to NULL and clean up memory if necessary.
    if (Archive::is_loading::value)
    {
      if (treeOwner && referenceTree)
        delete referenceTree;

      referenceTree = NULL;
      oldFromNewReferences.clear();
      treeOwner = false;
    }
  }
  else
  {
    // Delete the current reference tree, if necessary and if we are loading.
    if (Archive::is_loading::value)
    {
      if (treeOwner && referenceTree)
        delete referenceTree;

      // After we load the tree, we will own it.
      treeOwner = true;
    }

    ar & CreateNVP(referenceTree, "referenceTree");
    ar & CreateNVP(oldFromNewReferences, "oldFromNewReferences");

    // If we are loading, set the dataset accordingly and clean up memory if
    // necessary.
    if (Archive::is_loading::value)
    {
      if (setOwner && referenceSet)
        delete referenceSet;

      referenceSet = &referenceTree->Dataset();
      metric = referenceTree->Metric();
      setOwner = false;
    }
  }
}

} // namespace neighbor
} // namespace mlpack

#endif
