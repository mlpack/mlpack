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

// Construct the object.
template<typename SortPolicy,
         typename MetricType,
         typename TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, TreeType, TraversalType>::
NeighborSearch(const typename TreeType::Mat& referenceSetIn,
               const bool naive,
               const bool singleMode,
               const MetricType metric) :
    referenceTree(naive ? NULL :
        BuildTree<TreeType>(referenceSetIn, oldFromNewReferences)),
    referenceSet(naive ? referenceSetIn : referenceTree->Dataset()),
    treeOwner(!naive), // False if a tree was passed.  If naive, then no trees.
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    metric(metric),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

// Construct the object.
template<typename SortPolicy,
         typename MetricType,
         typename TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, TreeType, TraversalType>::
NeighborSearch(TreeType* referenceTree,
               const bool singleMode,
               const MetricType metric) :
    referenceTree(referenceTree),
    referenceSet(referenceTree->Dataset()),
    treeOwner(false),
    naive(false),
    singleMode(singleMode),
    metric(metric),
    baseCases(0),
    scores(0)
{
  // Nothing else to initialize.
}

// Clean memory.
template<typename SortPolicy,
         typename MetricType,
         typename TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, TreeType, TraversalType>::
    ~NeighborSearch()
{
  if (treeOwner && referenceTree)
    delete referenceTree;
}

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy,
         typename MetricType,
         typename TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, TreeType, TraversalType>::Search(
    const typename TreeType::Mat& querySet,
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

  // Mapping is only necessary if the tree rearranges points.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    if (!singleMode && !naive)
      distancePtr = new arma::mat; // Query indices need to be mapped.

    if (treeOwner)
      neighborPtr = new arma::Mat<size_t>; // All indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  neighborPtr->fill(size_t() - 1);
  distancePtr->set_size(k, querySet.n_cols);
  distancePtr->fill(SortPolicy::WorstDistance());

  typedef NeighborSearchRules<SortPolicy, MetricType, TreeType> RuleType;

  if (naive)
  {
    // Create the helper object for the tree traversal.
    RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr, metric);

    // The naive brute-force traversal.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet.n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases += querySet.n_cols * referenceSet.n_cols;
  }
  else if (singleMode)
  {
    // Create the helper object for the tree traversal.
    RuleType rules(referenceSet, querySet, *neighborPtr, *distancePtr, metric);

    // Create the traverser.
    typename TreeType::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";
  }
  else // Dual-tree recursion.
  {
    // Build the query tree.
    Timer::Stop("computing_neighbors");
    Timer::Start("tree_building");
    TreeType* queryTree = BuildTree<TreeType>(querySet, oldFromNewQueries);
    Timer::Stop("tree_building");
    Timer::Start("computing_neighbors");

    // Create the helper object for the tree traversal.
    RuleType rules(referenceSet, queryTree->Dataset(), *neighborPtr,
        *distancePtr, metric);

    // Create the traverser.
    TraversalType<RuleType> traverser(rules);

    traverser.Traverse(*queryTree, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";

    delete queryTree;
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
} // Search()

template<typename SortPolicy,
         typename MetricType,
         typename TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, TreeType, TraversalType>::Search(
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

  // Create the helper object for the traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, querySet, *neighborPtr, distances, metric);

  // Create the traverser.
  TraversalType<RuleType> traverser(rules);
  traverser.Traverse(*queryTree, *referenceTree);

  scores += rules.Scores();
  baseCases += rules.BaseCases();

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

template<typename SortPolicy,
         typename MetricType,
         typename TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, TreeType, TraversalType>::Search(
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

  // Create the helper object for the traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, TreeType> RuleType;
  RuleType rules(referenceSet, referenceSet, *neighborPtr, *distancePtr,
      metric, true /* don't return the same point as nearest neighbor */);

  if (naive)
  {
    // The naive brute-force solution.
    for (size_t i = 0; i < referenceSet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet.n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases += referenceSet.n_cols * referenceSet.n_cols;
  }
  else if (singleMode)
  {
    // Create the traverser.
    typename TreeType::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < referenceSet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";
  }
  else
  {
    // Create the traverser.
    TraversalType<RuleType> traverser(rules);

    traverser.Traverse(*referenceTree, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";
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

// Return a String of the Object.
template<typename SortPolicy,
         typename MetricType,
         typename TreeType,
         template<typename> class TraversalType>
std::string NeighborSearch<SortPolicy, MetricType, TreeType, TraversalType>::
    ToString() const
{
  std::ostringstream convert;
  convert << "NeighborSearch [" << this << "]" << std::endl;
  convert << "  Reference set: " << referenceSet.n_rows << "x" ;
  convert << referenceSet.n_cols << std::endl;
  if (referenceTree)
    convert << "  Reference tree: " << referenceTree << std::endl;
  convert << "  Tree owner: " << treeOwner << std::endl;
  convert << "  Naive: " << naive << std::endl;
  convert << "  Metric: " << std::endl;
  convert << mlpack::util::Indent(metric.ToString(),2);
  return convert.str();
}

}; // namespace neighbor
}; // namespace mlpack

#endif
