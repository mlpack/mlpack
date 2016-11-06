/**
 * @file range_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the RangeSearch class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_IMPL_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RANGE_SEARCH_IMPL_HPP

// Just in case it hasn't been included.
#include "range_search.hpp"

// The rules for traversal.
#include "range_search_rules.hpp"

namespace mlpack {
namespace range {

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

template<typename TreeType>
TreeType* BuildTree(
    typename TreeType::Mat&& dataset,
    std::vector<size_t>& oldFromNew,
    const typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  return new TreeType(std::move(dataset), oldFromNew);
}

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

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<MetricType, MatType, TreeType>::RangeSearch(
    const MatType& referenceSetIn,
    const bool naive,
    const bool singleMode,
    const MetricType metric) :
    referenceTree(naive ? NULL : BuildTree<Tree>(
        const_cast<MatType&>(referenceSetIn), oldFromNewReferences)),
    referenceSet(naive ? &referenceSetIn : &referenceTree->Dataset()),
    treeOwner(!naive), // If in naive mode, we are not building any trees.
    setOwner(false),
    naive(naive),
    singleMode(!naive && singleMode), // Naive overrides single mode.
    metric(metric),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

// Move constructor.
template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<MetricType, MatType, TreeType>::RangeSearch(
    MatType&& referenceSet,
    const bool naive,
    const bool singleMode,
    const MetricType metric) :
    referenceTree(naive ? NULL : BuildTree<Tree>(std::move(referenceSet),
        oldFromNewReferences)),
    referenceSet(naive ? new MatType(std::move(referenceSet)) :
        &referenceTree->Dataset()),
    treeOwner(!naive),
    setOwner(naive),
    naive(naive),
    singleMode(!naive && singleMode),
    metric(metric),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<MetricType, MatType, TreeType>::RangeSearch(
    Tree* referenceTree,
    const bool singleMode,
    const MetricType metric) :
    referenceTree(referenceTree),
    referenceSet(&referenceTree->Dataset()),
    treeOwner(false),
    setOwner(false),
    naive(false),
    singleMode(singleMode),
    metric(metric),
    baseCases(0),
    scores(0)
{
  // Nothing else to initialize.
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<MetricType, MatType, TreeType>::RangeSearch(
    const bool naive,
    const bool singleMode,
    const MetricType metric) :
    referenceTree(NULL),
    referenceSet(new MatType()), // Empty matrix.
    treeOwner(false),
    setOwner(true),
    naive(naive),
    singleMode(singleMode),
    metric(metric),
    baseCases(0),
    scores(0)
{
  // Build the tree on the empty dataset, if necessary.
  if (!naive)
  {
    referenceTree = BuildTree<Tree>(const_cast<MatType&>(*referenceSet),
        oldFromNewReferences);
    treeOwner = true;
  }
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<MetricType, MatType, TreeType>::~RangeSearch()
{
  if (treeOwner && referenceTree)
    delete referenceTree;
  if (setOwner && referenceSet)
    delete referenceSet;
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<MetricType, MatType, TreeType>::Train(
    const MatType& referenceSet)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
    delete referenceTree;

  // Rebuild the tree, if necessary.
  if (!naive)
  {
    referenceTree = BuildTree<Tree>(const_cast<MatType&>(referenceSet),
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
    this->referenceSet = &referenceTree->Dataset();
  else
    this->referenceSet = &referenceSet;
  setOwner = false;
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<MetricType, MatType, TreeType>::Train(
    MatType&& referenceSet)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
    delete referenceTree;

  // We may need to rebuild the tree.
  if (!naive)
  {
    referenceTree = BuildTree<Tree>(std::move(referenceSet),
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

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<MetricType, MatType, TreeType>::Train(
  Tree* referenceTree)
{
  if (naive)
    throw std::invalid_argument("cannot train on given reference tree when "
        "naive search (without trees) is desired");

  if (treeOwner && referenceTree)
    delete this->referenceTree;
  if (setOwner && referenceSet)
    delete this->referenceSet;

  this->referenceTree = referenceTree;
  this->referenceSet = &referenceTree->Dataset();
  treeOwner = false;
  setOwner = false;
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<MetricType, MatType, TreeType>::Search(
    const MatType& querySet,
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances)
{
  if (querySet.n_rows != referenceSet->n_rows)
  {
    std::ostringstream oss;
    oss << "RangeSearch::Search(): dimensionalities of query set ("
        << querySet.n_rows << ") and reference set (" << referenceSet->n_rows
        << ") do not match!";
    throw std::invalid_argument(oss.str());
  }

  Timer::Start("range_search/computing_neighbors");

  // This will hold mappings for query points, if necessary.
  std::vector<size_t> oldFromNewQueries;

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid extra copies, we will store the unmapped neighbors and distances
  // in a separate object.
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;
  std::vector<std::vector<double>>* distancePtr = &distances;

  // Mapping is only necessary if the tree rearranges points.
  if (tree::TreeTraits<Tree>::RearrangesDataset)
  {
    // Query indices only need to be mapped if we are building the query tree
    // ourselves.
    if (!singleMode && !naive)
    {
      distancePtr = new std::vector<std::vector<double>>;
      neighborPtr = new std::vector<std::vector<size_t>>;
    }

    // Reference indices only need to be mapped if we built the reference tree
    // ourselves.
    else if (treeOwner)
      neighborPtr = new std::vector<std::vector<size_t>>;
  }

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distancePtr->clear();
  distancePtr->resize(querySet.n_cols);

  // Create the helper object for the traversal.
  typedef RangeSearchRules<MetricType, Tree> RuleType;

  // Reset counts.
  baseCases = 0;
  scores = 0;

  if (naive)
  {
    RuleType rules(*referenceSet, querySet, range, *neighborPtr, *distancePtr,
        metric);

    // The naive brute-force solution.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet->n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases += (querySet.n_cols * referenceSet->n_cols);
  }
  else if (singleMode)
  {
    // Create the traverser.
    RuleType rules(*referenceSet, querySet, range, *neighborPtr, *distancePtr,
        metric);
    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    baseCases += rules.BaseCases();
    scores += rules.Scores();
  }
  else // Dual-tree recursion.
  {
    // Build the query tree.
    Timer::Stop("range_search/computing_neighbors");
    Timer::Start("range_search/tree_building");
    Tree* queryTree = BuildTree<Tree>(const_cast<MatType&>(querySet),
        oldFromNewQueries);
    Timer::Stop("range_search/tree_building");
    Timer::Start("range_search/computing_neighbors");

    // Create the traverser.
    RuleType rules(*referenceSet, queryTree->Dataset(), range, *neighborPtr,
        *distancePtr, metric);
    typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*queryTree, *referenceTree);

    baseCases += rules.BaseCases();
    scores += rules.Scores();

    // Clean up tree memory.
    delete queryTree;
  }

  Timer::Stop("range_search/computing_neighbors");

  // Map points back to original indices, if necessary.
  if (tree::TreeTraits<Tree>::RearrangesDataset)
  {
    if (!singleMode && !naive && treeOwner)
    {
      // We must map both query and reference indices.
      neighbors.clear();
      neighbors.resize(querySet.n_cols);
      distances.clear();
      distances.resize(querySet.n_cols);

      for (size_t i = 0; i < distances.size(); i++)
      {
        // Map distances (copy a column).
        const size_t queryMapping = oldFromNewQueries[i];
        distances[queryMapping] = (*distancePtr)[i];

        // Copy each neighbor individually, because we need to map it.
        neighbors[queryMapping].resize(distances[queryMapping].size());
        for (size_t j = 0; j < distances[queryMapping].size(); j++)
          neighbors[queryMapping][j] =
              oldFromNewReferences[(*neighborPtr)[i][j]];
      }

      // Finished with temporary objects.
      delete neighborPtr;
      delete distancePtr;
    }
    else if (!singleMode && !naive)
    {
      // We must map query indices only.
      neighbors.clear();
      neighbors.resize(querySet.n_cols);
      distances.clear();
      distances.resize(querySet.n_cols);

      for (size_t i = 0; i < distances.size(); ++i)
      {
        // Map distances and neighbors (copy a column).
        const size_t queryMapping = oldFromNewQueries[i];
        distances[queryMapping] = (*distancePtr)[i];
        neighbors[queryMapping] = (*neighborPtr)[i];
      }

      // Finished with temporary objects.
      delete neighborPtr;
      delete distancePtr;
    }
    else if (treeOwner)
    {
      // We must map reference indices only.
      neighbors.clear();
      neighbors.resize(querySet.n_cols);

      for (size_t i = 0; i < neighbors.size(); i++)
      {
        neighbors[i].resize((*neighborPtr)[i].size());
        for (size_t j = 0; j < neighbors[i].size(); j++)
          neighbors[i][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
      }

      // Finished with temporary object.
      delete neighborPtr;
    }
  }
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<MetricType, MatType, TreeType>::Search(
    Tree* queryTree,
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances)
{
  Timer::Start("range_search/computing_neighbors");

  // Get a reference to the query set.
  const MatType& querySet = queryTree->Dataset();

  // Make sure we are in dual-tree mode.
  if (singleMode || naive)
    throw std::invalid_argument("cannot call RangeSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  // We won't need to map query indices, but will we need to map distances?
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;

  if (treeOwner && tree::TreeTraits<Tree>::RearrangesDataset)
    neighborPtr = new std::vector<std::vector<size_t>>;

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distances.clear();
  distances.resize(querySet.n_cols);

  // Create the helper object for the traversal.
  typedef RangeSearchRules<MetricType, Tree> RuleType;
  RuleType rules(*referenceSet, queryTree->Dataset(), range, *neighborPtr,
      distances, metric);

  // Create the traverser.
  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*queryTree, *referenceTree);

  Timer::Stop("range_search/computing_neighbors");

  baseCases = rules.BaseCases();
  scores = rules.Scores();

  // Do we need to map indices?
  if (treeOwner && tree::TreeTraits<Tree>::RearrangesDataset)
  {
    // We must map reference indices only.
    neighbors.clear();
    neighbors.resize(querySet.n_cols);

    for (size_t i = 0; i < neighbors.size(); i++)
    {
      neighbors[i].resize((*neighborPtr)[i].size());
      for (size_t j = 0; j < neighbors[i].size(); j++)
        neighbors[i][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
    }

    // Finished with temporary object.
    delete neighborPtr;
  }
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<MetricType, MatType, TreeType>::Search(
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances)
{
  Timer::Start("range_search/computing_neighbors");

  // Here, we will use the query set as the reference set.
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;
  std::vector<std::vector<double>>* distancePtr = &distances;

  if (tree::TreeTraits<Tree>::RearrangesDataset && treeOwner)
  {
    // We will always need to rearrange in this case.
    distancePtr = new std::vector<std::vector<double>>;
    neighborPtr = new std::vector<std::vector<size_t>>;
  }

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(referenceSet->n_cols);
  distancePtr->clear();
  distancePtr->resize(referenceSet->n_cols);

  // Create the helper object for the traversal.
  typedef RangeSearchRules<MetricType, Tree> RuleType;
  RuleType rules(*referenceSet, *referenceSet, range, *neighborPtr,
      *distancePtr, metric, true /* don't return the query in the results */);

  if (naive)
  {
    // The naive brute-force solution.
    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      for (size_t j = 0; j < referenceSet->n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases = (referenceSet->n_cols * referenceSet->n_cols);
    scores = 0;
  }
  else if (singleMode)
  {
    // Create the traverser.
    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    baseCases = rules.BaseCases();
    scores = rules.Scores();
  }
  else // Dual-tree recursion.
  {
    // Create the traverser.
    typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*referenceTree, *referenceTree);

    baseCases = rules.BaseCases();
    scores = rules.Scores();
  }

  Timer::Stop("range_search/computing_neighbors");

  // Do we need to map the reference indices?
  if (treeOwner && tree::TreeTraits<Tree>::RearrangesDataset)
  {
    neighbors.clear();
    neighbors.resize(referenceSet->n_cols);
    distances.clear();
    distances.resize(referenceSet->n_cols);

    for (size_t i = 0; i < distances.size(); i++)
    {
      // Map distances (copy a column).
      const size_t refMapping = oldFromNewReferences[i];
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
}

template<typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
template<typename Archive>
void RangeSearch<MetricType, MatType, TreeType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  using data::CreateNVP;

  // Serialize preferences for search.
  ar & CreateNVP(naive, "naive");
  ar & CreateNVP(singleMode, "singleMode");

  // Reset base cases and scores if we are loading.
  if (Archive::is_loading::value)
  {
    baseCases = 0;
    scores = 0;
  }

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
      metric = referenceTree->Metric(); // Get the metric from the tree.
      setOwner = false;
    }
  }
}

} // namespace range
} // namespace mlpack

#endif
