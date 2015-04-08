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

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    const typename TreeType::Mat& referenceSetIn,
    const bool naive,
    const bool singleMode,
    const MetricType metric) :
    referenceSet((tree::TreeTraits<TreeType>::RearrangesDataset && !naive)
        ? referenceCopy : referenceSetIn),
    referenceTree(NULL),
    treeOwner(!naive), // If in naive mode, we are not building any trees.
    naive(naive),
    singleMode(!naive && singleMode), // Naive overrides single mode.
    metric(metric)
{
  // Build the trees.
  Timer::Start("range_search/tree_building");

  // If in naive mode, then we do not need to build trees.
  if (!naive)
  {
    // Copy the dataset, if it will be modified during tree building.
    if (tree::TreeTraits<TreeType>::RearrangesDataset)
      referenceCopy = referenceSetIn;

    // The const_cast is safe; if RearrangesDataset == false, then it'll be
    // casted back to const anyway, and if not, referenceSet points to
    // referenceCopy, which isn't const.
    referenceTree = BuildTree<TreeType>(
        const_cast<typename TreeType::Mat&>(referenceSet),
        oldFromNewReferences);
  }

  Timer::Stop("range_search/tree_building");
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::RangeSearch(
    TreeType* referenceTree,
    const bool singleMode,
    const MetricType metric) :
    referenceSet(referenceTree->Dataset()),
    referenceTree(referenceTree),
    treeOwner(false),
    naive(false),
    singleMode(singleMode),
    metric(metric)
{
  // Nothing else to initialize.
}

template<typename MetricType, typename TreeType>
RangeSearch<MetricType, TreeType>::~RangeSearch()
{
  if (treeOwner && referenceTree)
    delete referenceTree;
}

template<typename MetricType, typename TreeType>
void RangeSearch<MetricType, TreeType>::Search(
    const typename TreeType::Mat& querySet,
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances)
{
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
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    // Query indices only need to be mapped if we are building the query tree
    // ourselves.
    if (!singleMode && !naive)
      distancePtr = new std::vector<std::vector<double>>;

    // Reference indices only need to be mapped if we built the reference tree
    // ourselves.
    if (treeOwner)
      neighborPtr = new std::vector<std::vector<size_t>>;
  }

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distancePtr->clear();
  distancePtr->resize(querySet.n_cols);

  // Create the helper object for the traversal.
  typedef RangeSearchRules<MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, querySet, range, *neighborPtr, *distancePtr,
      metric);

  if (naive)
  {
    // The naive brute-force solution.
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
    // Build the query tree.
    Timer::Stop("range_search/computing_neighbors");
    Timer::Start("range_search/tree_building");
    typename TreeType::Mat queryCopy;
    if (tree::TreeTraits<TreeType>::RearrangesDataset)
      queryCopy = querySet;

    const typename TreeType::Mat& querySetRef =
        (tree::TreeTraits<TreeType>::RearrangesDataset) ? querySet : queryCopy;
    TreeType* queryTree = BuildTree<TreeType>(
        const_cast<typename TreeType::Mat&>(querySetRef), oldFromNewQueries);
    Timer::Stop("range_search/tree_building");
    Timer::Start("range_search/computing_neighbors");

    // Create the traverser.
    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*queryTree, *referenceTree);

    // Clean up tree memory.
    delete queryTree;
  }

  Timer::Stop("range_search/computing_neighbors");

  // Map points back to original indices, if necessary.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
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

template<typename MetricType, typename TreeType>
void RangeSearch<MetricType, TreeType>::Search(
    TreeType* queryTree,
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances)
{
  Timer::Start("range_search/computing_neighbors");

  // Get a reference to the query set.
  const typename TreeType::Mat& querySet = queryTree->Dataset();

  // Make sure we are in dual-tree mode.
  if (singleMode || naive)
    throw std::invalid_argument("cannot call RangeSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  // We won't need to map query indices, but will we need to map distances?
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;

  if (treeOwner && tree::TreeTraits<TreeType>::RearrangesDataset)
    neighborPtr = new std::vector<std::vector<size_t>>;

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distances.clear();
  distances.resize(querySet.n_cols);

  // Create the helper object for the traversal.
  typedef RangeSearchRules<MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, queryTree->Dataset(), range, *neighborPtr,
      distances, metric);

  // Create the traverser.
  typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*queryTree, *referenceTree);

  Timer::Stop("range_search/computing_neighbors");

  // Do we need to map indices?
  if (treeOwner && tree::TreeTraits<TreeType>::RearrangesDataset)
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

template<typename MetricType, typename TreeType>
void RangeSearch<MetricType, TreeType>::Search(
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances)
{
  Timer::Start("range_search/computing_neighbors");

  // Here, we will use the query set as the reference set.
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;
  std::vector<std::vector<double>>* distancePtr = &distances;

  if (tree::TreeTraits<TreeType>::RearrangesDataset && treeOwner)
  {
    // We will always need to rearrange in this case.
    distancePtr = new std::vector<std::vector<double>>;
    neighborPtr = new std::vector<std::vector<size_t>>;
  }

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(referenceSet.n_cols);
  distancePtr->clear();
  distancePtr->resize(referenceSet.n_cols);

  // Create the helper object for the traversal.
  typedef RangeSearchRules<MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, referenceSet, range, *neighborPtr, *distancePtr,
      metric);

  if (naive)
  {
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
  else // Dual-tree recursion.
  {
    // Create the traverser.
    typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*referenceTree, *referenceTree);
  }

  Timer::Stop("range_search/computing_neighbors");

  // Do we need to map the reference indices?
  if (treeOwner && tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    neighbors.clear();
    neighbors.resize(referenceSet.n_cols);
    distances.clear();
    distances.resize(referenceSet.n_cols);

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
}

template<typename MetricType, typename TreeType>
std::string RangeSearch<MetricType, TreeType>::ToString() const
{
  std::ostringstream convert;
  convert << "Range Search  [" << this << "]" << std::endl;
  if (treeOwner)
    convert << "  Tree Owner: TRUE" << std::endl;
  if (naive)
    convert << "  Naive: TRUE" << std::endl;
  convert << "  Metric: " << std::endl <<
      mlpack::util::Indent(metric.ToString(),2);
  return convert.str();
}

}; // namespace range
}; // namespace mlpack

#endif
