/**
 * @file methods/neighbor_search/neighbor_search_impl.hpp
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

#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/greedy_single_tree_traverser.hpp>
#include "neighbor_search_rules.hpp"
#include <mlpack/core/tree/spill_tree/is_spill_tree.hpp>

namespace mlpack {

// Construct the object.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
    DualTreeTraversalType, SingleTreeTraversalType>::
NeighborSearch(MatType referenceSetIn,
               const NeighborSearchMode mode,
               const double epsilon,
               const DistanceType distance) :
    referenceTree(mode == NAIVE_MODE ? NULL :
        BuildTree<Tree>(std::move(referenceSetIn), oldFromNewReferences)),
    referenceSet(mode == NAIVE_MODE ?  new MatType(std::move(referenceSetIn)) :
        &referenceTree->Dataset()),
    searchMode(mode),
    epsilon(epsilon),
    distance(distance),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");
}

// Construct the object.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
    DualTreeTraversalType, SingleTreeTraversalType>::
NeighborSearch(Tree referenceTree,
               const NeighborSearchMode mode,
               const double epsilon,
               const DistanceType distance) :
    referenceTree(new Tree(std::move(referenceTree))),
    referenceSet(&this->referenceTree->Dataset()),
    searchMode(mode),
    epsilon(epsilon),
    distance(distance),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");
}

// Construct the object without a reference dataset.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
    DualTreeTraversalType, SingleTreeTraversalType>::
NeighborSearch(const NeighborSearchMode mode,
               const double epsilon,
               const DistanceType distance) :
    referenceTree(NULL),
    referenceSet(mode == NAIVE_MODE ? new MatType() : NULL), // Empty matrix.
    searchMode(mode),
    epsilon(epsilon),
    distance(distance),
    baseCases(0),
    scores(0),
    treeNeedsReset(false)
{
  if (epsilon < 0)
    throw std::invalid_argument("epsilon must be non-negative");

  // Build the tree on the empty dataset, if necessary.
  if (mode != NAIVE_MODE)
  {
    referenceTree = BuildTree<Tree>(std::move(MatType()),
        oldFromNewReferences);
    referenceSet = &referenceTree->Dataset();
  }
}

// Copy constructor.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
    DualTreeTraversalType, SingleTreeTraversalType>::
NeighborSearch(const NeighborSearch& other) :
    oldFromNewReferences(other.oldFromNewReferences),
    referenceTree(other.referenceTree ? new Tree(*other.referenceTree) : NULL),
    referenceSet(other.referenceTree ? &referenceTree->Dataset() :
        new MatType(*other.referenceSet)),
    searchMode(other.searchMode),
    epsilon(other.epsilon),
    distance(other.distance),
    baseCases(other.baseCases),
    scores(other.scores),
    treeNeedsReset(false)
{
  // Nothing else to do.
}

// Move constructor.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
    DualTreeTraversalType, SingleTreeTraversalType>::
NeighborSearch(NeighborSearch&& other) :
    oldFromNewReferences(std::move(other.oldFromNewReferences)),
    referenceTree(other.referenceTree),
    referenceSet(other.referenceSet),
    searchMode(other.searchMode),
    epsilon(other.epsilon),
    distance(std::move(other.distance)),
    baseCases(other.baseCases),
    scores(other.scores),
    treeNeedsReset(other.treeNeedsReset)
{
  // Clear the other model.
  other.referenceTree = BuildTree<Tree>(std::move(MatType()),
      other.oldFromNewReferences);
  other.referenceSet = &other.referenceTree->Dataset();
  other.searchMode = DUAL_TREE_MODE,
  other.epsilon = 0.0;
  other.baseCases = 0;
  other.scores = 0;
  other.treeNeedsReset = false;
}

// Copy operator.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy,
               DistanceType,
               MatType,
               TreeType,
               DualTreeTraversalType,
               SingleTreeTraversalType>&
NeighborSearch<SortPolicy,
               DistanceType,
               MatType,
               TreeType,
               DualTreeTraversalType,
               SingleTreeTraversalType>::operator=(const NeighborSearch& other)
{
  if (&other == this)
    return *this; // Nothing to do.

  // Clean memory first.
  if (referenceTree)
    delete referenceTree;
  else
    delete referenceSet;

  oldFromNewReferences = other.oldFromNewReferences;
  referenceTree = other.referenceTree ? new Tree(*other.referenceTree) : NULL;
  referenceSet = other.referenceTree ? &referenceTree->Dataset() :
      new MatType(*other.referenceSet);
  searchMode = other.searchMode;
  epsilon = other.epsilon;
  distance = other.distance;
  baseCases = other.baseCases;
  scores = other.scores;
  treeNeedsReset = false;
}

// Move operator.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy,
               DistanceType,
               MatType,
               TreeType,
               DualTreeTraversalType,
               SingleTreeTraversalType>&
NeighborSearch<SortPolicy,
               DistanceType,
               MatType,
               TreeType,
               DualTreeTraversalType,
               SingleTreeTraversalType>::operator=(NeighborSearch&& other)
{
  if (&other == this)
    return *this; // Nothing to do.

  // Clean memory first.
  if (referenceTree)
    delete referenceTree;
  else
    delete referenceSet;

  oldFromNewReferences = std::move(other.oldFromNewReferences);
  referenceTree = other.referenceTree;
  referenceSet = other.referenceSet;
  searchMode = other.searchMode;
  epsilon = other.epsilon;
  distance = other.distance;
  baseCases = other.baseCases;
  scores = other.scores;
  treeNeedsReset = other.treeNeedsReset;

  // Reset the other object.  Clean memory if needed.
  if (!other.referenceTree)
    delete other.referenceSet;

  other.referenceTree = BuildTree<Tree>(std::move(MatType()),
      other.oldFromNewReferences);
  other.referenceSet = &other.referenceTree->Dataset();
  other.searchMode = DUAL_TREE_MODE,
  other.epsilon = 0.0;
  other.baseCases = 0;
  other.scores = 0;
  other.treeNeedsReset = false;
}

// Clean memory.
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
    DualTreeTraversalType, SingleTreeTraversalType>::~NeighborSearch()
{
  if (referenceTree)
    delete referenceTree;
  else
    delete referenceSet;
}

template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
    DualTreeTraversalType, SingleTreeTraversalType>::
Train(MatType referenceSetIn)
{
  // Clean up the old tree, if we built one.
  if (referenceTree)
  {
    oldFromNewReferences.clear();
    delete referenceTree;
    referenceTree = NULL;
  }
  else
  {
    delete referenceSet;
  }

  // We may need to rebuild the tree.
  if (searchMode != NAIVE_MODE)
  {
    referenceTree = BuildTree<Tree>(std::move(referenceSetIn),
        oldFromNewReferences);
    referenceSet = &referenceTree->Dataset();
  }
  else
  {
    referenceSet = new MatType(std::move(referenceSetIn));
  }
}

template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Train(Tree referenceTree)
{
  if (searchMode == NAIVE_MODE)
    throw std::invalid_argument("cannot train on given reference tree when "
        "naive search (without trees) is desired");

  if (this->referenceTree)
  {
    oldFromNewReferences.clear();
    delete this->referenceTree;
  }
  else
  {
    delete this->referenceSet;
  }

  this->referenceTree = new Tree(std::move(referenceTree));
  this->referenceSet = &this->referenceTree->Dataset();
}

/**
 * Computes the best neighbors and stores them in resultingNeighbors and
 * distances.
 */
template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
template<typename IndexType>
void NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Search(
    const MatType& querySet,
    const size_t k,
    arma::Mat<IndexType>& neighbors,
    arma::Mat<ElemType>& distances)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "Requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }

  baseCases = 0;
  scores = 0;

  // This will hold mappings for query points, if necessary.
  std::vector<size_t> oldFromNewQueries;

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid an extra copy, we will store the neighbors and distances in a
  // separate matrix.
  arma::Mat<IndexType>* neighborPtr = &neighbors;
  arma::Mat<ElemType>* distancePtr = &distances;

  // Mapping is only necessary if the tree rearranges points.
  if (TreeTraits<Tree>::RearrangesDataset)
  {
    if (searchMode == DUAL_TREE_MODE)
    {
      distancePtr = new arma::Mat<ElemType>; // Query indices need to be mapped.
      neighborPtr = new arma::Mat<IndexType>;
    }
    else if (!oldFromNewReferences.empty())
      neighborPtr = new arma::Mat<IndexType>; // Reference indices need mapping.
  }

  // Set the size of the neighbor and distance matrices.
  neighborPtr->set_size(k, querySet.n_cols);
  distancePtr->set_size(k, querySet.n_cols);

  using RuleType = NeighborSearchRules<SortPolicy, DistanceType, Tree>;

  switch (searchMode)
  {
    case NAIVE_MODE:
    {
      // Create the helper object for the tree traversal.
      RuleType rules(*referenceSet, querySet, k, distance, epsilon);

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
      RuleType rules(*referenceSet, querySet, k, distance, epsilon);

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
      Tree* queryTree = BuildTree<Tree>(querySet, oldFromNewQueries);

      // Create the helper object for the tree traversal.
      RuleType rules(*referenceSet, queryTree->Dataset(), k, distance, epsilon);

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
      RuleType rules(*referenceSet, querySet, k, distance);

      // Create the traverser.
      GreedySingleTreeTraverser<Tree, RuleType> traverser(rules);

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

  // Map points back to original indices, if necessary.
  if (TreeTraits<Tree>::RearrangesDataset)
  {
    if (searchMode == DUAL_TREE_MODE && !oldFromNewReferences.empty())
    {
      // We must map both query and reference indices.
      neighbors.set_size(k, querySet.n_cols);
      distances.set_size(k, querySet.n_cols);

      for (size_t i = 0; i < distances.n_cols; ++i)
      {
        // Map distances (copy a column).
        distances.col(oldFromNewQueries[i]) = distancePtr->col(i);

        // Map indices of neighbors.
        for (size_t j = 0; j < distances.n_rows; ++j)
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
      for (size_t i = 0; i < neighbors.n_cols; ++i)
        for (size_t j = 0; j < neighbors.n_rows; ++j)
          neighbors(j, i) = oldFromNewReferences[(*neighborPtr)(j, i)];

      // Finished with temporary matrix.
      delete neighborPtr;
    }
  }
} // Search()

template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
template<typename IndexType>
void NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Search(
    Tree& queryTree,
    const size_t k,
    arma::Mat<IndexType>& neighbors,
    arma::Mat<ElemType>& distances,
    bool sameSet)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "Requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }

  // Make sure we are in dual-tree mode.
  if (searchMode != DUAL_TREE_MODE)
    throw std::invalid_argument("cannot call NeighborSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  baseCases = 0;
  scores = 0;

  // Get a reference to the query set.
  const MatType& querySet = queryTree.Dataset();

  // We won't need to map query indices, but will we need to map distances?
  arma::Mat<IndexType>* neighborPtr = &neighbors;

  if (!oldFromNewReferences.empty() && TreeTraits<Tree>::RearrangesDataset)
    neighborPtr = new arma::Mat<IndexType>;

  neighborPtr->set_size(k, querySet.n_cols);
  distances.set_size(k, querySet.n_cols);

  // Create the helper object for the traversal.
  using RuleType = NeighborSearchRules<SortPolicy, DistanceType, Tree>;
  RuleType rules(*referenceSet, querySet, k, distance, epsilon, sameSet);

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

  // Do we need to map indices?
  if (!oldFromNewReferences.empty() && TreeTraits<Tree>::RearrangesDataset)
  {
    // We must map reference indices only.
    neighbors.set_size(k, querySet.n_cols);

    // Map indices of neighbors.
    for (size_t i = 0; i < neighbors.n_cols; ++i)
      for (size_t j = 0; j < neighbors.n_rows; ++j)
        neighbors(j, i) = oldFromNewReferences[(*neighborPtr)(j, i)];

    // Finished with temporary matrix.
    delete neighborPtr;
  }
}

template<typename SortPolicy,
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
template<typename IndexType>
void NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Search(
    const size_t k,
    arma::Mat<IndexType>& neighbors,
    arma::Mat<ElemType>& distances)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "Requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }
  if (k == referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "Requested value of k (" << k << ") is equal to the number of "
        << "points in the reference set (" << referenceSet->n_cols << ") and "
        << "no query set has been provided.";
    throw std::invalid_argument(ss.str());
  }

  baseCases = 0;
  scores = 0;

  arma::Mat<IndexType>* neighborPtr = &neighbors;
  arma::Mat<ElemType>* distancePtr = &distances;

  if (!oldFromNewReferences.empty() && TreeTraits<Tree>::RearrangesDataset)
  {
    // We will always need to rearrange in this case.
    distancePtr = new MatType;
    neighborPtr = new arma::Mat<IndexType>;
  }

  // Initialize results.
  neighborPtr->set_size(k, referenceSet->n_cols);
  distancePtr->set_size(k, referenceSet->n_cols);

  // Create the helper object for the traversal.
  using RuleType = NeighborSearchRules<SortPolicy, DistanceType, Tree>;
  RuleType rules(*referenceSet, *referenceSet, k, distance, epsilon,
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

      if (IsSpillTree<Tree>::value)
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
      GreedySingleTreeTraverser<Tree, RuleType> traverser(rules);

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

  // Do we need to map the reference indices?
  if (!oldFromNewReferences.empty() && TreeTraits<Tree>::RearrangesDataset)
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
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
double NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::EffectiveError(
    arma::Mat<ElemType>& foundDistances,
    arma::Mat<ElemType>& realDistances)
{
  if (foundDistances.n_rows != realDistances.n_rows ||
      foundDistances.n_cols != realDistances.n_cols)
    throw std::invalid_argument("matrices provided must have equal size");

  double effectiveError = 0;
  size_t numCases = 0;

  for (size_t i = 0; i < foundDistances.n_elem; ++i)
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
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
template<typename IndexType>
double NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::Recall(
    arma::Mat<IndexType>& foundNeighbors,
    arma::Mat<IndexType>& realNeighbors)
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
         typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
template<typename Archive>
void NeighborSearch<SortPolicy, DistanceType, MatType, TreeType,
DualTreeTraversalType, SingleTreeTraversalType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // Serialize preferences for search.
  ar(CEREAL_NVP(searchMode));
  ar(CEREAL_NVP(treeNeedsReset));

  // If we are doing naive search, we serialize the dataset.  Otherwise we
  // serialize the tree.
  if (searchMode == NAIVE_MODE)
  {
    // Delete the current reference set, if necessary and if we are loading.
    if (cereal::is_loading<Archive>() && referenceSet)
    {
      delete referenceSet;
    }

    ar(CEREAL_POINTER(const_cast<MatType*&>(referenceSet)));
    ar(CEREAL_NVP(distance));

    // If we are loading, set the tree to NULL and clean up memory if necessary.
    if (cereal::is_loading<Archive>())
    {
      if (referenceTree)
        delete referenceTree;

      referenceTree = NULL;
      oldFromNewReferences.clear();
    }
  }
  else
  {
    // Delete the current reference tree, if necessary and if we are loading.
    if (cereal::is_loading<Archive>() && referenceTree)
    {
      delete referenceTree;
    }

    ar(CEREAL_POINTER(referenceTree));
    ar(CEREAL_NVP(oldFromNewReferences));

    // If we are loading, set the dataset accordingly and clean up memory if
    // necessary.
    if (cereal::is_loading<Archive>())
    {
      referenceSet = &referenceTree->Dataset();
      distance = referenceTree->Distance(); // Get the distance from the tree.
    }
  }

  // Reset base cases and scores.
  if (cereal::is_loading<Archive>())
  {
    baseCases = 0;
    scores = 0;
  }
}

} // namespace mlpack

#endif
