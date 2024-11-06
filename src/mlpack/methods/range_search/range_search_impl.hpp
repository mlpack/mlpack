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
    const bool naive,
    const bool singleMode,
    const DistanceType distance) :
    referenceTree(naive ? NULL : BuildTree<Tree>(std::move(referenceSet),
        oldFromNewReferences)),
    referenceSet(naive ? new MatType(std::move(referenceSet)) :
        &referenceTree->Dataset()),
    treeOwner(!naive),
    naive(naive),
    singleMode(!naive && singleMode),
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
    Tree* referenceTree,
    const bool singleMode,
    const DistanceType distance) :
    referenceTree(referenceTree),
    referenceSet(&referenceTree->Dataset()),
    treeOwner(false),
    naive(false),
    singleMode(singleMode),
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
    const bool naive,
    const bool singleMode,
    const DistanceType distance) :
    referenceTree(NULL),
    referenceSet(naive ? new MatType() : NULL), // Empty matrix.
    treeOwner(false),
    naive(naive),
    singleMode(singleMode),
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
    naive = other.naive;
    singleMode = other.singleMode;
    distance = std::move(other.distance);
    baseCases = other.baseCases;
    scores = other.scores;

    // Clear other object.
    other.referenceTree = nullptr;
    other.referenceSet = nullptr;
    other.treeOwner = false;
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
  if (treeOwner && referenceTree)
    delete referenceTree;
  if (naive && referenceSet)
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
  if (naive && this->referenceSet)
    delete this->referenceSet;

  if (!naive)
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
  Tree* referenceTree)
{
  if (naive)
    throw std::invalid_argument("cannot train on given reference tree when "
        "naive search (without trees) is desired");

  // Can only train when passed argument `referenceTree` is not nullptr.
  if (treeOwner && referenceTree)
  {
    delete this->referenceTree;

    this->referenceTree = referenceTree;
    this->referenceSet = &referenceTree->Dataset();
    treeOwner = false;
  }
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
    if (!singleMode && !naive)
    {
      distancePtr = new std::vector<std::vector<ElemType>>;
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
  using RuleType = RangeSearchRules<DistanceType, Tree>;

  // Reset counts.
  baseCases = 0;
  scores = 0;

  if (naive)
  {
    RuleType rules(*referenceSet, querySet, range, *neighborPtr, *distancePtr,
        distance);

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
    if (!singleMode && !naive && treeOwner)
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
    Tree* queryTree,
    const RangeType<ElemType>& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<ElemType>>& distances)
{
  // If there are no points, there is no search to be done.
  if (referenceSet->n_cols == 0)
    return;

  // Get a reference to the query set.
  const MatType& querySet = queryTree->Dataset();

  // Make sure we are in dual-tree mode.
  if (singleMode || naive)
    throw std::invalid_argument("cannot call RangeSearch::Search() with a "
        "query tree when naive or singleMode are set to true");

  // We won't need to map query indices, but will we need to map distances?
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;

  if (treeOwner && TreeTraits<Tree>::RearrangesDataset)
    neighborPtr = new std::vector<std::vector<size_t>>;

  // Resize each vector.
  neighborPtr->clear(); // Just in case there was anything in it.
  neighborPtr->resize(querySet.n_cols);
  distances.clear();
  distances.resize(querySet.n_cols);

  // Create the helper object for the traversal.
  using RuleType = RangeSearchRules<DistanceType, Tree>;
  RuleType rules(*referenceSet, queryTree->Dataset(), range, *neighborPtr,
      distances, distance);

  // Create the traverser.
  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*queryTree, *referenceTree);

  baseCases = rules.BaseCases();
  scores = rules.Scores();

  // Do we need to map indices?
  if (treeOwner && TreeTraits<Tree>::RearrangesDataset)
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
  // If there are no points, there is no search to be done.
  if (referenceSet->n_cols == 0)
    return;

  // Here, we will use the query set as the reference set.
  std::vector<std::vector<size_t>>* neighborPtr = &neighbors;
  std::vector<std::vector<ElemType>>* distancePtr = &distances;

  if (TreeTraits<Tree>::RearrangesDataset && treeOwner)
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

  // Do we need to map the reference indices?
  if (treeOwner && TreeTraits<Tree>::RearrangesDataset)
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
    Archive& ar, const uint32_t /* version */)
{
  // Serialize preferences for search.
  ar(CEREAL_NVP(naive));
  ar(CEREAL_NVP(singleMode));

  // Reset base cases and scores if we are loading.
  if (cereal::is_loading<Archive>())
  {
    baseCases = 0;
    scores = 0;
  }

  // If we are doing naive search, we serialize the dataset.  Otherwise we
  // serialize the tree.
  if (naive)
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

} // namespace mlpack

#endif
