/**
 * @file methods/range_search/range_search_impl.hpp
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

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(
    MatType referenceSet,
    const TreeSearchStrategy strategyIn,
    const DistanceType distance) :
    referenceTree((strategyIn == NAIVE) ? NULL :
        BuildTree<Tree>(std::move(referenceSet), oldFromNewReferences)),
    referenceSet((strategyIn == NAIVE) ? new MatType(std::move(referenceSet)) :
        &referenceTree->Dataset()),
    treeOwner(strategyIn != NAIVE),
    strategy(strategyIn),
    needsSync(false),
    naive(strategyIn == NAIVE),
    singleMode(strategyIn == SINGLE_TREE),
    distance(distance),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

// Deprecated and will be removed in mlpack 5.0.0.
template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(
    MatType referenceSet,
    const bool naiveIn,
    const bool singleModeIn,
    const DistanceType distance) :
    referenceTree(naiveIn ? NULL : BuildTree<Tree>(std::move(referenceSet),
        oldFromNewReferences)),
    referenceSet(naiveIn ? new MatType(std::move(referenceSet)) :
        &referenceTree->Dataset()),
    treeOwner(!naiveIn),
    strategy(naiveIn ? NAIVE : (singleModeIn ? SINGLE_TREE : DUAL_TREE)),
    needsSync(false),
    naive(naiveIn),
    singleMode(!naiveIn && singleModeIn),
    distance(distance),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(
    Tree referenceTreeIn,
    const TreeSearchStrategy strategyIn,
    const DistanceType distance) :
    referenceTree(new Tree(std::move(referenceTreeIn))),
    referenceSet(&referenceTree->Dataset()),
    treeOwner(true),
    strategy(strategyIn),
    needsSync(false),
    naive(false),
    singleMode(strategyIn == SINGLE_TREE),
    distance(distance),
    baseCases(0),
    scores(0)
{
  // Nothing else to initialize.
}

// Deprecated and will be removed in mlpack 5.0.0.
template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(
    Tree* referenceTree,
    const bool singleModeIn,
    const DistanceType distance) :
    referenceTree(referenceTree),
    referenceSet(&referenceTree->Dataset()),
    treeOwner(false),
    strategy(singleModeIn ? SINGLE_TREE : DUAL_TREE),
    needsSync(false),
    naive(false),
    singleMode(singleModeIn),
    distance(distance),
    baseCases(0),
    scores(0)
{
  // Nothing else to initialize.
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(
    const TreeSearchStrategy strategyIn,
    const DistanceType distance) :
    referenceTree(NULL),
    referenceSet((strategyIn == NAIVE) ? new MatType() : NULL), // Empty matrix.
    treeOwner(false),
    strategy(strategyIn),
    needsSync(false),
    naive(strategyIn == NAIVE),
    singleMode(strategyIn == SINGLE_TREE),
    distance(distance),
    baseCases(0),
    scores(0)
{
  // Build the tree on the empty dataset, if necessary.
  if (strategy != NAIVE)
  {
    referenceTree = BuildTree<Tree>(std::move(MatType()),
        oldFromNewReferences);
    referenceSet = &referenceTree->Dataset();
    treeOwner = true;
  }
}

// Deprecated and will be removed in mlpack 5.0.0.
template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(
    const bool naiveIn,
    const bool singleModeIn,
    const DistanceType distance) :
    referenceTree(NULL),
    referenceSet(naiveIn ? new MatType() : NULL), // Empty matrix.
    treeOwner(false),
    strategy(naiveIn ? NAIVE : (singleModeIn ? SINGLE_TREE : DUAL_TREE)),
    needsSync(false),
    naive(naiveIn),
    singleMode(singleModeIn),
    distance(distance),
    baseCases(0),
    scores(0)
{
  // Build the tree on the empty dataset, if necessary.
  if (!naive)
  {
    referenceTree = BuildTree<Tree>(std::move(MatType()),
        oldFromNewReferences);
    referenceSet = &referenceTree->Dataset();
    treeOwner = true;
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(
    const RangeSearch& other) :
    oldFromNewReferences(other.oldFromNewReferences),
    referenceTree(other.referenceTree ? new Tree(*other.referenceTree) : NULL),
    referenceSet(other.referenceTree ? &referenceTree->Dataset() :
        new MatType(*other.referenceSet)),
    treeOwner(other.referenceTree),
    strategy(other.strategy),
    needsSync(other.needsSync),
    naive(other.naive),
    singleMode(other.singleMode),
    distance(other.distance),
    baseCases(other.baseCases),
    scores(other.scores)
{
  // Nothing to do.
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::RangeSearch(RangeSearch&& other) :
    oldFromNewReferences(std::move(other.oldFromNewReferences)),
    referenceTree(other.referenceTree),
    referenceSet(other.referenceSet),
    treeOwner(other.treeOwner),
    strategy(other.strategy),
    needsSync(other.needsSync),
    naive(other.naive),
    singleMode(other.singleMode),
    distance(std::move(other.distance)),
    baseCases(other.baseCases),
    scores(other.scores)
{
  // Clear other object.
  other.referenceTree =
      BuildTree<Tree>(std::move(MatType()), other.oldFromNewReferences);
  other.referenceSet = &other.referenceTree->Dataset();
  other.treeOwner = true;
  other.strategy = DUAL_TREE;
  other.needsSync = false;
  other.naive = false;
  other.singleMode = false;
  other.baseCases = 0;
  other.scores = 0;
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>&
RangeSearch<DistanceType, MatType, TreeType>::operator=(
    const RangeSearch& other)
{
  if (this != &other)
  {
    oldFromNewReferences = other.oldFromNewReferences;
    referenceTree = other.referenceTree ? new Tree(*other.referenceTree) :
        nullptr;
    referenceSet = other.referenceTree ? &referenceTree->Dataset() :
        new MatType(*other.referenceSet);
    treeOwner = other.referenceTree;
    strategy = other.strategy;
    needsSync = other.needsSync;
    naive = other.naive;
    singleMode = other.singleMode;
    distance = other.distance;
    baseCases = other.baseCases;
    scores = other.scores;
  }
  return *this;
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>&
RangeSearch<DistanceType, MatType, TreeType>::operator=(RangeSearch&& other)
{
  if (this != &other)
  {
    // Clean memory first.
    if (treeOwner)
      delete referenceTree;
    if (naive)
      delete referenceSet;

    // Move the other model.
    oldFromNewReferences = std::move(other.oldFromNewReferences);
    referenceTree = other.referenceTree;
    referenceSet = other.referenceSet;
    treeOwner = other.treeOwner;
    strategy = other.strategy;
    needsSync = other.needsSync;
    naive = other.naive;
    singleMode = other.singleMode;
    distance = std::move(other.distance);
    baseCases = other.baseCases;
    scores = other.scores;

    // Clear other object.
    other.referenceTree = nullptr;
    other.referenceSet = nullptr;
    other.treeOwner = false;
    other.strategy = DUAL_TREE;
    other.needsSync = false;
    other.naive = false;
    other.singleMode = false;
    other.baseCases = 0;
    other.scores = 0;
  }
  return *this;
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
RangeSearch<DistanceType, MatType, TreeType>::~RangeSearch()
{
  SyncStrategy();

  if (treeOwner && referenceTree)
    delete referenceTree;
  if ((strategy == NAIVE) && referenceSet)
    delete referenceSet;
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<DistanceType, MatType, TreeType>::Train(
    MatType referenceSet)
{
  SyncStrategy();

  // Clean up the old tree, if we built one.
  if (treeOwner && referenceTree)
    delete referenceTree;

  // We may need to rebuild the tree.
  if (strategy != NAIVE)
  {
    referenceTree = BuildTree<Tree>(std::move(referenceSet),
        oldFromNewReferences);
    treeOwner = true;
  }
  else
  {
    referenceTree = NULL;
    treeOwner = false;
  }

  // Delete the old reference set, if we owned it.
  if ((strategy == NAIVE) && this->referenceSet)
    delete this->referenceSet;

  if (strategy != NAIVE)
  {
    this->referenceSet = &referenceTree->Dataset();
  }
  else
  {
    this->referenceSet = new MatType(std::move(referenceSet));
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<DistanceType, MatType, TreeType>::Train(
    Tree referenceTree)
{
  SyncStrategy();

  if (strategy == NAIVE)
    throw std::invalid_argument("cannot train on given reference tree when "
        "naive search (without trees) is desired");

  // Can only train when passed argument `referenceTree` is not nullptr.
  if (treeOwner)
    delete this->referenceTree;

  this->referenceTree = new Tree(std::move(referenceTree));
  this->oldFromNewReferences.clear();
  this->referenceSet = &this->referenceTree->Dataset();
  treeOwner = true;
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<DistanceType, MatType, TreeType>::Train(
    Tree* referenceTree)
{
  Train(*referenceTree);
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<DistanceType, MatType, TreeType>::Search(
    const MatType& querySet,
    const RangeType<ElemType>& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<ElemType>>& distances)
{
  SyncStrategy();

  // Make sure the strategy is valid.
  if (strategy == GREEDY_SINGLE_TREE)
  {
    throw std::invalid_argument("RangeSearch::Search(): GREEDY_SINGLE_TREE "
        "search strategy not supported; use DUAL_TREE, SINGLE_TREE, or NAIVE "
        "instead!");
  }

  util::CheckSameDimensionality(querySet, *referenceSet,
      "RangeSearch::Search()", "query set");

  // If there are no points, there is no search to be done.
  if (referenceSet->n_cols == 0)
    return;

  // This will hold mappings for query points, if necessary.
  std::vector<size_t> oldFromNewQueries;

  // If we have built the trees ourselves, then we will have to map all the
  // indices back to their original indices when this computation is finished.
  // To avoid extra copies, we will store the unmapped neighbors and distances
  // in a separate object.
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;
  std::vector<std::vector<ElemType>>* distancePtr = &distances;

  // Mapping is only necessary if the tree rearranges points.
  if (TreeTraits<Tree>::RearrangesDataset)
  {
    // Query indices only need to be mapped if we are building the query tree
    // ourselves.
    if (strategy == DUAL_TREE)
    {
      distancePtr = new std::vector<std::vector<ElemType>>;
      neighborPtr = new std::vector<std::vector<size_t>>;
    }

    // Reference indices only need to be mapped if we built the reference tree
    // ourselves.
    else if (treeOwner && oldFromNewReferences.size() > 0)
      neighborPtr = new std::vector<std::vector<size_t>>;
  }

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distancePtr->clear();
  distancePtr->resize(querySet.n_cols);

  // Create the helper object for the traversal.
  using RuleType = RangeSearchRules<DistanceType, Tree>;

  // Reset counts.
  baseCases = 0;
  scores = 0;

  if (strategy == NAIVE)
  {
    RuleType rules(*referenceSet, querySet, range, *neighborPtr, *distancePtr,
        distance);

    // The naive brute-force solution.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      for (size_t j = 0; j < referenceSet->n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases += (querySet.n_cols * referenceSet->n_cols);
  }
  else if (strategy == SINGLE_TREE)
  {
    // Create the traverser.
    RuleType rules(*referenceSet, querySet, range, *neighborPtr, *distancePtr,
        distance);
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
    Tree* queryTree = BuildTree<Tree>(querySet, oldFromNewQueries);

    // Create the traverser.
    RuleType rules(*referenceSet, queryTree->Dataset(), range, *neighborPtr,
        *distancePtr, distance);
    typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

    traverser.Traverse(*queryTree, *referenceTree);

    baseCases += rules.BaseCases();
    scores += rules.Scores();

    // Clean up tree memory.
    delete queryTree;
  }

  // Map points back to original indices, if necessary.
  if (TreeTraits<Tree>::RearrangesDataset)
  {
    if ((strategy == DUAL_TREE) && treeOwner && oldFromNewReferences.size() > 0)
    {
      // We must map both query and reference indices.
      neighbors.clear();
      neighbors.resize(querySet.n_cols);
      distances.clear();
      distances.resize(querySet.n_cols);

      for (size_t i = 0; i < distances.size(); ++i)
      {
        // Map distances (copy a column).
        const size_t queryMapping = oldFromNewQueries[i];
        distances[queryMapping] = (*distancePtr)[i];

        // Copy each neighbor individually, because we need to map it.
        neighbors[queryMapping].resize(distances[queryMapping].size());
        for (size_t j = 0; j < distances[queryMapping].size(); ++j)
          neighbors[queryMapping][j] =
              oldFromNewReferences[(*neighborPtr)[i][j]];
      }

      // Finished with temporary objects.
      delete neighborPtr;
      delete distancePtr;
    }
    else if (strategy == DUAL_TREE)
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
    else if (treeOwner && oldFromNewReferences.size() > 0)
    {
      // We must map reference indices only.
      neighbors.clear();
      neighbors.resize(querySet.n_cols);

      for (size_t i = 0; i < neighbors.size(); ++i)
      {
        neighbors[i].resize((*neighborPtr)[i].size());
        for (size_t j = 0; j < neighbors[i].size(); ++j)
          neighbors[i][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
      }

      // Finished with temporary object.
      delete neighborPtr;
    }
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<DistanceType, MatType, TreeType>::Search(
    Tree& queryTree,
    const RangeType<ElemType>& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<ElemType>>& distances,
    bool sameSet)
{
  SyncStrategy();

  // Make sure the strategy is valid.
  if (strategy == GREEDY_SINGLE_TREE)
  {
    throw std::invalid_argument("RangeSearch::Search(): GREEDY_SINGLE_TREE "
        "search strategy not supported; use DUAL_TREE, SINGLE_TREE, or NAIVE "
        "instead!");
  }

  // If there are no points, there is no search to be done.
  if (referenceSet->n_cols == 0)
    return;

  // Get a reference to the query set.
  const MatType& querySet = queryTree.Dataset();

  // Make sure we are in dual-tree mode.
  if (strategy != DUAL_TREE)
    throw std::invalid_argument("cannot call RangeSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  // We won't need to map query indices, but will we need to map distances?
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;

  if (treeOwner && TreeTraits<Tree>::RearrangesDataset &&
      oldFromNewReferences.size() > 0)
    neighborPtr = new std::vector<std::vector<size_t>>;

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distances.clear();
  distances.resize(querySet.n_cols);

  // Create the helper object for the traversal.
  using RuleType = RangeSearchRules<DistanceType, Tree>;
  RuleType rules(*referenceSet, queryTree.Dataset(), range, *neighborPtr,
      distances, distance, sameSet);

  // Create the traverser.
  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(queryTree, *referenceTree);

  baseCases = rules.BaseCases();
  scores = rules.Scores();

  // Do we need to map indices?
  if (treeOwner && TreeTraits<Tree>::RearrangesDataset &&
      oldFromNewReferences.size() > 0)
  {
    // We must map reference indices only.
    neighbors.clear();
    neighbors.resize(querySet.n_cols);

    for (size_t i = 0; i < neighbors.size(); ++i)
    {
      neighbors[i].resize((*neighborPtr)[i].size());
      for (size_t j = 0; j < neighbors[i].size(); ++j)
        neighbors[i][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
    }

    // Finished with temporary object.
    delete neighborPtr;
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<DistanceType, MatType, TreeType>::Search(
    const RangeType<ElemType>& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<ElemType>>& distances)
{
  SyncStrategy();

  // If there are no points, there is no search to be done.
  if (referenceSet->n_cols == 0)
    return;

  // Make sure the strategy is valid.
  if (strategy == GREEDY_SINGLE_TREE)
  {
    throw std::invalid_argument("RangeSearch::Search(): GREEDY_SINGLE_TREE "
        "search strategy not supported; use DUAL_TREE, SINGLE_TREE, or NAIVE "
        "instead!");
  }

  // Here, we will use the query set as the reference set.
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;
  std::vector<std::vector<ElemType>>* distancePtr = &distances;

  if (TreeTraits<Tree>::RearrangesDataset && treeOwner &&
      oldFromNewReferences.size() > 0)
  {
    // We will always need to rearrange in this case.
    distancePtr = new std::vector<std::vector<ElemType>>;
    neighborPtr = new std::vector<std::vector<size_t>>;
  }

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(referenceSet->n_cols);
  distancePtr->clear();
  distancePtr->resize(referenceSet->n_cols);

  // Create the helper object for the traversal.
  using RuleType = RangeSearchRules<DistanceType, Tree>;
  RuleType rules(*referenceSet, *referenceSet, range, *neighborPtr,
      *distancePtr, distance, true /* don't return the query in the results */);

  if (strategy == NAIVE)
  {
    // The naive brute-force solution.
    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      for (size_t j = 0; j < referenceSet->n_cols; ++j)
        rules.BaseCase(i, j);

    baseCases = (referenceSet->n_cols * referenceSet->n_cols);
    scores = 0;
  }
  else if (strategy == SINGLE_TREE)
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

  // Do we need to map the reference indices?
  if (treeOwner && TreeTraits<Tree>::RearrangesDataset &&
      oldFromNewReferences.size() > 0)
  {
    neighbors.clear();
    neighbors.resize(referenceSet->n_cols);
    distances.clear();
    distances.resize(referenceSet->n_cols);

    for (size_t i = 0; i < distances.size(); ++i)
    {
      // Map distances (copy a column).
      const size_t refMapping = oldFromNewReferences[i];
      distances[refMapping] = (*distancePtr)[i];

      // Copy each neighbor individually, because we need to map it.
      neighbors[refMapping].resize(distances[refMapping].size());
      for (size_t j = 0; j < distances[refMapping].size(); ++j)
      {
        neighbors[refMapping][j] = oldFromNewReferences[(*neighborPtr)[i][j]];
      }
    }

    // Finished with temporary objects.
    delete neighborPtr;
    delete distancePtr;
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
template<typename Archive>
void RangeSearch<DistanceType, MatType, TreeType>::serialize(
    Archive& ar, const uint32_t version)
{
  SyncStrategy();

  // Serialize preferences for search.
  if (version >= 1)
  {
    ar(CEREAL_NVP(strategy));
    // This is necessary for reverse compatibility and can be removed in mlpack
    // 5.0.0.
    naive = (strategy == NAIVE);
    singleMode = (strategy == SINGLE_TREE);
    needsSync = false;
  }
  else
  {
    // Older versions stored the strategy as two booleans.
    ar(CEREAL_NVP(naive));
    ar(CEREAL_NVP(singleMode));
    strategy = singleMode ? SINGLE_TREE : (naive ? NAIVE : DUAL_TREE);
    needsSync = false;
  }

  // Reset base cases and scores if we are loading.
  if (cereal::is_loading<Archive>())
  {
    baseCases = 0;
    scores = 0;
  }

  // If we are doing naive search, we serialize the dataset.  Otherwise we
  // serialize the tree.
  if (strategy == NAIVE)
  {
    if (cereal::is_loading<Archive>())
    {
      if (referenceSet)
        delete referenceSet;
    }

    ar(CEREAL_POINTER(const_cast<MatType*&>(referenceSet)));
    ar(CEREAL_NVP(distance));

    // If we are loading, set the tree to NULL and clean up memory if necessary.
    if (cereal::is_loading<Archive>())
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
    if (cereal::is_loading<Archive>())
    {
      if (treeOwner && referenceTree)
        delete referenceTree;

      // After we load the tree, we will own it.
      treeOwner = true;
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
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RangeSearch<DistanceType, MatType, TreeType>::SyncStrategy()
{
  if (needsSync)
  {
    // If the user has called Naive() or SingleMode() most recently, then we
    // take those preferences.
    strategy = singleMode ? SINGLE_TREE : (naive ? NAIVE : DUAL_TREE);
    needsSync = false;
  }
}

} // namespace mlpack

#endif
