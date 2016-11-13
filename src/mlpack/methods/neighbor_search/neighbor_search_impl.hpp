/**
 * @file neighbor_search_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Neighbor-Search class to perform all-nearest-neighbors on
 * two specified data sets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_IMPL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/tree/greedy_single_tree_traverser.hpp>
#include "neighbor_search_rules.hpp"
#include <mlpack/core/tree/spill_tree/is_spill_tree.hpp>

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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, DualTreeTraversalType,
SingleTreeTraversalType>::NeighborSearch(const MatType& referenceSetIn,
                                         const NeighborSearchMode mode,
                                         const double epsilon,
                                         const MetricType metric) :
    referenceTree(mode == NAIVE_MODE ? NULL :
        BuildTree<MatType, Tree>(referenceSetIn, oldFromNewReferences)),
    referenceSet(mode == NAIVE_MODE ? &referenceSetIn :
        &referenceTree->Dataset()),
    treeOwner(mode != NAIVE_MODE),
    setOwner(false),
    searchMode(mode),
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, DualTreeTraversalType,
SingleTreeTraversalType>::NeighborSearch(MatType&& referenceSetIn,
                                         const NeighborSearchMode mode,
                                         const double epsilon,
                                         const MetricType metric) :
    referenceTree(mode == NAIVE_MODE ? NULL :
        BuildTree<MatType, Tree>(std::move(referenceSetIn),
                                 oldFromNewReferences)),
    referenceSet(mode == NAIVE_MODE ? new MatType(std::move(referenceSetIn)) :
        &referenceTree->Dataset()),
    treeOwner(mode != NAIVE_MODE),
    setOwner(mode == NAIVE_MODE),
    searchMode(mode),
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, DualTreeTraversalType,
SingleTreeTraversalType>::NeighborSearch(const Tree& referenceTree,
                                         const NeighborSearchMode mode,
                                         const double epsilon,
                                         const MetricType metric) :
    referenceTree(new Tree(referenceTree)),
    referenceSet(&this->referenceTree->Dataset()),
    treeOwner(true),
    setOwner(false),
    searchMode(mode),
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, DualTreeTraversalType,
SingleTreeTraversalType>::NeighborSearch(Tree&& referenceTree,
                                         const NeighborSearchMode mode,
                                         const double epsilon,
                                         const MetricType metric) :
    referenceTree(new Tree(std::move(referenceTree))),
    referenceSet(&this->referenceTree->Dataset()),
    treeOwner(true),
    setOwner(false),
    searchMode(mode),
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, DualTreeTraversalType,
SingleTreeTraversalType>::NeighborSearch(const NeighborSearchMode mode,
                                         const double epsilon,
                                         const MetricType metric) :
    referenceTree(NULL),
    referenceSet(new MatType()), // Empty matrix.
    treeOwner(false),
    setOwner(true),
    searchMode(mode),
    epsilon(epsilon),
    metric(metric),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");

  // Build the tree on the empty dataset, if necessary.
  if (mode != NAIVE_MODE)
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, MetricType, MatType, TreeType, DualTreeTraversalType,
SingleTreeTraversalType>::~NeighborSearch()
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Train(
    const MatType& referenceSet)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
  {
    oldFromNewReferences.clear();
    delete referenceTree;
  }

  // We may need to rebuild the tree.
  if (searchMode != NAIVE_MODE)
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

  if (searchMode != NAIVE_MODE)
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Train(MatType&& referenceSetIn)
{
  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
  {
    oldFromNewReferences.clear();
    delete referenceTree;
  }

  // We may need to rebuild the tree.
  if (searchMode != NAIVE_MODE)
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

  if (searchMode != NAIVE_MODE)
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Train(
    const Tree& referenceTree)
{
  if (searchMode == NAIVE_MODE)
    throw std::invalid_argument("cannot train on given reference tree when "
        "naive search (without trees) is desired");

  if (treeOwner && this->referenceTree)
  {
    oldFromNewReferences.clear();
    delete this->referenceTree;
  }

  if (setOwner && referenceSet)
    delete this->referenceSet;

  this->referenceTree = new Tree(referenceTree);
  this->referenceSet = &this->referenceTree->Dataset();
  treeOwner = true;
  setOwner = false;
}

template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Train(Tree&& referenceTree)
{
  if (searchMode == NAIVE_MODE)
    throw std::invalid_argument("cannot train on given reference tree when "
        "naive search (without trees) is desired");

  if (treeOwner && this->referenceTree)
  {
    oldFromNewReferences.clear();
    delete this->referenceTree;
  }

  if (setOwner && referenceSet)
    delete this->referenceSet;

  this->referenceTree = new Tree(std::move(referenceTree));
  this->referenceSet = &this->referenceTree->Dataset();
  treeOwner = true;
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Search(
    const MatType& querySet,
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
    if (searchMode == DUAL_TREE_MODE)
    {
      distancePtr = new arma::mat; // Query indices need to be mapped.
      neighborPtr = new arma::Mat<size_t>;
    }
    else if (!oldFromNewReferences.empty())
      neighborPtr = new arma::Mat<size_t>; // Reference indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);

  typedef NeighborSearchRules<SortPolicy, MetricType, Tree> RuleType;

  switch(searchMode)
  {
    case NAIVE_MODE:
    {
      // Create the helper object for the tree traversal.
      RuleType rules(*referenceSet, querySet, k, metric, epsilon);

      // The naive brute-force traversal.
      for (size_t i = 0; i < querySet.n_cols; ++i)
        for (size_t j = 0; j < referenceSet->n_cols; ++j)
          rules.BaseCase(i, j);

      baseCases += querySet.n_cols * referenceSet->n_cols;

      rules.GetResults(*neighborPtr, *distancePtr);
      break;
    }
    case SINGLE_TREE_MODE:
    {
      // Create the helper object for the tree traversal.
      RuleType rules(*referenceSet, querySet, k, metric, epsilon);

      // Create the traverser.
      SingleTreeTraversalType<RuleType> traverser(rules);

      // Now have it traverse for each point.
      for (size_t i = 0; i < querySet.n_cols; ++i)
        traverser.Traverse(i, *referenceTree);

      scores += rules.Scores();
      baseCases += rules.BaseCases();

      Log::Info << rules.Scores() << " node combinations were scored."
          << std::endl;
      Log::Info << rules.BaseCases() << " base cases were calculated."
          << std::endl;

      rules.GetResults(*neighborPtr, *distancePtr);
      break;
    }
    case DUAL_TREE_MODE:
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
      DualTreeTraversalType<RuleType> traverser(rules);

      traverser.Traverse(*queryTree, *referenceTree);

      scores += rules.Scores();
      baseCases += rules.BaseCases();

      Log::Info << rules.Scores() << " node combinations were scored."
          << std::endl;
      Log::Info << rules.BaseCases() << " base cases were calculated."
          << std::endl;

      rules.GetResults(*neighborPtr, *distancePtr);

      delete queryTree;
      break;
    }
    case GREEDY_SINGLE_TREE_MODE:
    {
      // Create the helper object for the tree traversal.
      RuleType rules(*referenceSet, querySet, k, metric);

      // Create the traverser.
      tree::GreedySingleTreeTraverser<Tree, RuleType> traverser(rules);

      // Now have it traverse for each point.
      for (size_t i = 0; i < querySet.n_cols; ++i)
        traverser.Traverse(i, *referenceTree);

      scores += rules.Scores();
      baseCases += rules.BaseCases();

      Log::Info << rules.Scores() << " node combinations were scored."
          << std::endl;
      Log::Info << rules.BaseCases() << " base cases were calculated."
          << std::endl;

      rules.GetResults(*neighborPtr, *distancePtr);
      break;
    }
  }

  Timer::Stop("computing_neighbors");

  // Map points back to original indices, if necessary.
  if (tree::TreeTraits<Tree>::RearrangesDataset)
  {
    if (searchMode == DUAL_TREE_MODE && !oldFromNewReferences.empty())
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
    else if (searchMode == DUAL_TREE_MODE)
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
    else if (!oldFromNewReferences.empty())
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Search(
    Tree& queryTree,
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
  if (searchMode != DUAL_TREE_MODE)
    throw std::invalid_argument("cannot call NeighborSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  Timer::Start("computing_neighbors");

  baseCases = 0;
  scores = 0;

  // Get a reference to the query set.
  const MatType& querySet = queryTree.Dataset();

  // We won't need to map query indices, but will we need to map distances?
  arma::Mat<size_t>* neighborPtr = &neighbors;

  if (!oldFromNewReferences.empty() &&
      tree::TreeTraits<Tree>::RearrangesDataset)
    neighborPtr = new arma::Mat<size_t>;

  neighborPtr->set_size(k, querySet.n_cols);
  distances.set_size(k, querySet.n_cols);

  // Create the helper object for the traversal.
  typedef NeighborSearchRules<SortPolicy, MetricType, Tree> RuleType;
  RuleType rules(*referenceSet, querySet, k, metric, epsilon, sameSet);

  // Create the traverser.
  DualTreeTraversalType<RuleType> traverser(rules);
  traverser.Traverse(queryTree, *referenceTree);

  scores += rules.Scores();
  baseCases += rules.BaseCases();

  Log::Info << rules.Scores() << " node combinations were scored." << std::endl;
  Log::Info << rules.BaseCases() << " base cases were calculated." << std::endl;

  rules.GetResults(*neighborPtr, distances);

  Log::Info << rules.Scores() << " node combinations were scored.\n";
  Log::Info << rules.BaseCases() << " base cases were calculated.\n";

  Timer::Stop("computing_neighbors");

  // Do we need to map indices?
  if (!oldFromNewReferences.empty() &&
      tree::TreeTraits<Tree>::RearrangesDataset)
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
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Search(
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

  arma::Mat<size_t>* neighborPtr = &neighbors;
  arma::mat* distancePtr = &distances;

  if (!oldFromNewReferences.empty() &&
      tree::TreeTraits<Tree>::RearrangesDataset)
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

  switch (searchMode)
  {
    case NAIVE_MODE:
    {
      // The naive brute-force solution.
      for (size_t i = 0; i < referenceSet->n_cols; ++i)
        for (size_t j = 0; j < referenceSet->n_cols; ++j)
          rules.BaseCase(i, j);

      baseCases += referenceSet->n_cols * referenceSet->n_cols;
      break;
    }
    case SINGLE_TREE_MODE:
    {
      // Create the traverser.
      SingleTreeTraversalType<RuleType> traverser(rules);

      // Now have it traverse for each point.
      for (size_t i = 0; i < referenceSet->n_cols; ++i)
        traverser.Traverse(i, *referenceTree);

      scores += rules.Scores();
      baseCases += rules.BaseCases();

      Log::Info << rules.Scores() << " node combinations were scored."
          << std::endl;
      Log::Info << rules.BaseCases() << " base cases were calculated."
          << std::endl;
      break;
    }
    case DUAL_TREE_MODE:
    {
      // The dual-tree monochromatic search case may require resetting the
      // bounds in the tree.
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
      DualTreeTraversalType<RuleType> traverser(rules);

      if (tree::IsSpillTree<Tree>::value)
      {
        // For Dual Tree Search on SpillTree, the queryTree must be built with
        // non overlapping (tau = 0).
        Tree queryTree(*referenceSet);
        traverser.Traverse(queryTree, *referenceTree);
      }
      else
      {
        traverser.Traverse(*referenceTree, *referenceTree);
        // Next time we perform this search, we'll need to reset the tree.
        treeNeedsReset = true;
      }

      scores += rules.Scores();
      baseCases += rules.BaseCases();

      Log::Info << rules.Scores() << " node combinations were scored."
          << std::endl;
      Log::Info << rules.BaseCases() << " base cases were calculated."
          << std::endl;

      // Next time we perform this search, we'll need to reset the tree.
      treeNeedsReset = true;
      break;
    }
    case GREEDY_SINGLE_TREE_MODE:
    {
      // Create the traverser.
      tree::GreedySingleTreeTraverser<Tree, RuleType> traverser(rules);

      // Now have it traverse for each point.
      for (size_t i = 0; i < referenceSet->n_cols; ++i)
        traverser.Traverse(i, *referenceTree);

      scores += rules.Scores();
      baseCases += rules.BaseCases();

      Log::Info << rules.Scores() << " node combinations were scored."
          << std::endl;
      Log::Info << rules.BaseCases() << " base cases were calculated."
          << std::endl;
      break;
    }
  }

  rules.GetResults(*neighborPtr, *distancePtr);

  Timer::Stop("computing_neighbors");

  // Do we need to map the reference indices?
  if (!oldFromNewReferences.empty() &&
      tree::TreeTraits<Tree>::RearrangesDataset)
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

//! Calculate the average relative error.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
double NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::EffectiveError(
    arma::mat& foundDistances,
    arma::mat& realDistances)
{
  if (foundDistances.n_rows != realDistances.n_rows ||
      foundDistances.n_cols != realDistances.n_cols)
    throw std::invalid_argument("matrices provided must have equal size");

  double effectiveError = 0;
  size_t numCases = 0;

  for (size_t i = 0; i < foundDistances.n_elem; i++)
  {
    if (realDistances(i) != 0 &&
        foundDistances(i) != SortPolicy::WorstDistance())
    {
      effectiveError += fabs(foundDistances(i) - realDistances(i)) /
          realDistances(i);
      numCases++;
    }
  }

  if (numCases)
    effectiveError /= numCases;

  return effectiveError;
}

//! Calculate the recall.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
double NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Recall(
    arma::Mat<size_t>& foundNeighbors,
    arma::Mat<size_t>& realNeighbors)
{
  if (foundNeighbors.n_rows != realNeighbors.n_rows ||
      foundNeighbors.n_cols != realNeighbors.n_cols)
    throw std::invalid_argument("matrices provided must have equal size");

  size_t found = 0;
  for (size_t col = 0; col < foundNeighbors.n_cols; ++col)
    for (size_t row = 0; row < foundNeighbors.n_rows; ++row)
      for (size_t nei = 0; nei < realNeighbors.n_rows; ++nei)
        if (foundNeighbors(row, col) == realNeighbors(nei, col))
        {
          found++;
          break;
        }

  return ((double) found) / realNeighbors.n_elem;
}

//! Serialize the NeighborSearch model.
template<typename SortPolicy,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
template<typename Archive>
void NeighborSearch<SortPolicy, MetricType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  using data::CreateNVP;

  // Serialize preferences for search.
  ar & CreateNVP(searchMode, "searchMode");
  ar & CreateNVP(treeNeedsReset, "treeNeedsReset");

  // If we are doing naive search, we serialize the dataset.  Otherwise we
  // serialize the tree.
  if (searchMode == NAIVE_MODE)
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
