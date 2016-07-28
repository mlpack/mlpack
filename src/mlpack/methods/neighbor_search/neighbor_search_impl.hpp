/**
 * @file neighbor_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Neighbor-Search class to perform all-nearest-neighbors on
 * two specified data sets.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>

#include "neighbor_search_rules.hpp"

namespace mlpack {
namespace neighbor {

//! Call the tree constructor that does mapping.
template<typename MatType, typename TreeType>
TreeType* BuildTree(
    const MatType& dataset,
    std::vector<size_t>& oldFromNew,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  return new TreeType(dataset, oldFromNew);
}

//! Call the tree constructor that does not do mapping.
template<typename MatType, typename TreeType>
TreeType* BuildTree(
    const MatType& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == false, TreeType*
    >::type = 0)
{
  return new TreeType(dataset);
}

//! Call the tree construct that does mapping.
template<typename MatType, typename TreeType>
TreeType* BuildTree(
    MatType&& dataset,
    std::vector<size_t>& oldFromNew,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  return new TreeType(std::move(dataset), oldFromNew);
}

//! Call the tree constructor that does not do mapping.
template<typename MatType, typename TreeType>
TreeType* BuildTree(
    MatType&& dataset,
    std::vector<size_t>& /* oldFromNew */,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == false, TreeType*
    >::type = 0)
{
  return new TreeType(std::move(dataset));
}

// Construct the object.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
NeighborSearch(const MatType& referenceSetIn,
               const bool naive,
               const bool singleMode,
               const double epsilon,
               const MetricType metric) :
    referenceTree(naive ? NULL :
        BuildTree<MatType, Tree>(referenceSetIn, oldFromNewReferences)),
    referenceSet(naive ? &referenceSetIn : &referenceTree->Dataset()),
    treeOwner(!naive), // False if a tree was passed.  If naive, then no trees.
    setOwner(false),
    naive(naive),
    singleMode(!naive && singleMode), // No single mode if naive.
    epsilon(epsilon),
    metric(metric),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");
}

// Construct the object.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
NeighborSearch(MatType&& referenceSetIn,
               const bool naive,
               const bool singleMode,
               const double epsilon,
               const MetricType metric) :
    referenceTree(naive ? NULL :
        BuildTree<MatType, Tree>(std::move(referenceSetIn),
                                 oldFromNewReferences)),
    referenceSet(naive ? new MatType(std::move(referenceSetIn)) :
        &referenceTree->Dataset()),
    treeOwner(!naive),
    setOwner(naive),
    naive(naive),
    singleMode(!naive && singleMode),
    epsilon(epsilon),
    metric(metric),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");
}

// Construct the object.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
NeighborSearch(Tree* referenceTree,
               const bool singleMode,
               const double epsilon,
               const MetricType metric) :
    referenceTree(referenceTree),
    referenceSet(&referenceTree->Dataset()),
    treeOwner(false),
    setOwner(false),
    naive(false),
    singleMode(singleMode),
    epsilon(epsilon),
    metric(metric),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");
}

// Construct the object without a reference dataset.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
    NeighborSearch(const bool naive,
                   const bool singleMode,
                   const double epsilon,
                   const MetricType metric) :
    referenceTree(NULL),
    referenceSet(new MatType()), // Empty matrix.
    treeOwner(false),
    setOwner(true),
    naive(naive),
    singleMode(singleMode),
    epsilon(epsilon),
    metric(metric),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");
  // Build the tree on the empty dataset, if necessary.
  if (!naive)
  {
    referenceTree = BuildTree<MatType, Tree>(*referenceSet,
        oldFromNewReferences);
    treeOwner = true;
  }
}

// Clean memory.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
~NeighborSearch()
{
  if (treeOwner && referenceTree)
    delete referenceTree;
  if (setOwner && referenceSet)
    delete referenceSet;
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
Train(const MatType& referenceSet)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
    delete referenceTree;

  // We may need to rebuild the tree.
  if (!naive)
  {
    referenceTree = BuildTree<MatType, Tree>(referenceSet,
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
  setOwner = false; // We don't own the set in either case.
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
Train(MatType&& referenceSetIn)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
    delete referenceTree;

  // We may need to rebuild the tree.
  if (!naive)
  {
    referenceTree = BuildTree<MatType, Tree>(std::move(referenceSetIn),
        oldFromNewReferences);
    treeOwner = true;
  }
  else
  {
    treeOwner = false;
  }

  // Delete the old reference set, if we owned it.
  if (setOwner && referenceSet)
    delete referenceSet;

  if (!naive)
  {
    referenceSet = &referenceTree->Dataset();
    setOwner = false;
  }
  else
  {
    referenceSet = new MatType(std::move(referenceSetIn));
    setOwner = true;
  }
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
Train(Tree* referenceTree)
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

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
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

  baseCases = 0;
  scores = 0;

  // This will hold mappings for query points, if necessary.
  std::vector<size_t> oldFromNewQueries;

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid an extra copy, we will store the neighbors and distances in a
  // separate matrix.
  arma::Mat<size_t>* neighborPtr = &neighbors;
  arma::mat* distancePtr = &distances;

  // Mapping is only necessary if the tree rearranges points.
  if (tree::TreeTraits<Tree>::RearrangesDataset)
  {
    if (!singleMode && !naive)
    {
      distancePtr = new arma::mat; // Query indices need to be mapped.
      neighborPtr = new arma::Mat<size_t>;
    }
    else if (treeOwner)
      neighborPtr = new arma::Mat<size_t>; // Reference indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);

  typedef NeighborSearchRules<SortPolicy, MetricType, Tree> RuleType;

  if (naive)
  {
    // Create the helper object for the tree traversal.
    RuleType rules(*referenceSet, querySet, k, metric, epsilon);

    // The naive brute-force traversal.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet->n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases += querySet.n_cols * referenceSet->n_cols;

    rules.GetResults(*neighborPtr, *distancePtr);
  }
  else if (singleMode)
  {
    // Create the helper object for the tree traversal.
    RuleType rules(*referenceSet, querySet, k, metric, epsilon);

    // Create the traverser.
    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";

    rules.GetResults(*neighborPtr, *distancePtr);
  }
  else // Dual-tree recursion.
  {
    // Build the query tree.
    Timer::Stop("computing_neighbors");
    Timer::Start("tree_building");
    Tree* queryTree = BuildTree<MatType, Tree>(querySet, oldFromNewQueries);
    Timer::Stop("tree_building");
    Timer::Start("computing_neighbors");

    // Create the helper object for the tree traversal.
    RuleType rules(*referenceSet, queryTree->Dataset(), k, metric, epsilon);

    // Create the traverser.
    TraversalType<RuleType> traverser(rules);

    traverser.Traverse(*queryTree, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";

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
} // Search()

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
Search(Tree* queryTree,
       const size_t k,
       arma::Mat<size_t>& neighbors,
       arma::mat& distances,
       bool sameSet)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }

  // Make sure we are in dual-tree mode.
  if (singleMode || naive)
    throw std::invalid_argument("cannot call NeighborSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  Timer::Start("computing_neighbors");

  baseCases = 0;
  scores = 0;

  // Get a reference to the query set.
  const MatType& querySet = queryTree->Dataset();

  // We won't need to map query indices, but will we need to map distances?
  arma::Mat<size_t>* neighborPtr = &neighbors;

  if (treeOwner && tree::TreeTraits<Tree>::RearrangesDataset)
    neighborPtr = new arma::Mat<size_t>;

  neighborPtr->set_size(k, querySet.n_cols);
  distances.set_size(k, querySet.n_cols);

  // Create the helper object for the traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, Tree> RuleType;
  RuleType rules(*referenceSet, querySet, k, metric, epsilon, sameSet);

  // Create the traverser.
  TraversalType<RuleType> traverser(rules);
  traverser.Traverse(*queryTree, *referenceTree);

  scores += rules.Scores();
  baseCases += rules.BaseCases();

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
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
Search(const size_t k,
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

  baseCases = 0;
  scores = 0;

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

  // Create the helper object for the traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, Tree> RuleType;
  RuleType rules(*referenceSet, *referenceSet, k, metric, epsilon,
      true /* don't return the same point as nearest neighbor */);

  if (naive)
  {
    // The naive brute-force solution.
    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      for (size_t j = 0; j < referenceSet->n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases += referenceSet->n_cols * referenceSet->n_cols;
  }
  else if (singleMode)
  {
    // Create the traverser.
    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    // Now have it traverse for each point.
    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";
  }
  else
  {
    // The dual-tree monochromatic search case may require resetting the bounds
    // in the tree.
    if (treeNeedsReset)
    {
      std::stack<Tree*> nodes;
      nodes.push(referenceTree);
      while (!nodes.empty())
      {
        Tree* node = nodes.top();
        nodes.pop();

        // Reset bounds of this node.
        node->Stat().Reset();

        // Then add the children.
        for (size_t i = 0; i < node->NumChildren(); ++i)
          nodes.push(&node->Child(i));
      }
    }

    // Create the traverser.
    TraversalType<RuleType> traverser(rules);

    traverser.Traverse(*referenceTree, *referenceTree);

    scores += rules.Scores();
    baseCases += rules.BaseCases();

    Log::Info << rules.Scores() << " node combinations were scored.\n";
    Log::Info << rules.BaseCases() << " base cases were calculated.\n";

    // Next time we perform this search, we'll need to reset the tree.
    treeNeedsReset = true;
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

//! Serialize the NeighborSearch model.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class TraversalType>
template<typename Archive>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType, TraversalType>::
    Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  // Serialize preferences for search.
  ar & CreateNVP(naive, "naive");
  ar & CreateNVP(singleMode, "singleMode");
  ar & CreateNVP(treeNeedsReset, "treeNeedsReset");

  // If we are doing naive search, we serialize the dataset.  Otherwise we
  // serialize the tree.
  if (naive)
  {
    // Delete the current reference set, if necessary and if we are loading.
    if (Archive::is_loading::value)
    {
      if (setOwner && referenceSet)
        delete referenceSet;

      setOwner = true; // We will own the reference set when we load it.
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

  // Reset base cases and scores.
  if (Archive::is_loading::value)
  {
    baseCases = 0;
    scores = 0;
  }
}

} // namespace neighbor
} // namespace mlpack

#endif
