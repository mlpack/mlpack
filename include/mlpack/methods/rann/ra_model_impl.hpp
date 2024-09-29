/**
 * @file methods/rann/ra_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the RAModel class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_MODEL_IMPL_HPP
#define MLPACK_METHODS_RANN_RA_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "ra_model.hpp"
#include <mlpack/core/math/random_basis.hpp>

namespace mlpack {

inline RAModel::RAModel(const TreeTypes treeType, const bool randomBasis) :
    treeType(treeType),
    leafSize(20),
    randomBasis(randomBasis),
    raSearch(NULL)
{
  // Nothing to do.
}

// Copy constructor.
inline RAModel::RAModel(const RAModel& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(other.q),
    raSearch(other.raSearch->Clone())
{
  // Nothing to do.
}

// Move constructor.
inline RAModel::RAModel(RAModel&& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(std::move(other.q)),
    raSearch(std::move(other.raSearch))
{
  // Clear other model.
  other.treeType = TreeTypes::KD_TREE;
  other.leafSize = 20;
  other.randomBasis = false;
}

// Copy operator.
inline RAModel& RAModel::operator=(const RAModel& other)
{
  if (this != &other)
  {
    // Clear current model.
    delete raSearch;

    treeType = other.treeType;
    leafSize = other.leafSize;
    randomBasis = other.randomBasis;
    q = other.q;
    raSearch = other.raSearch->Clone();
  }

  return *this;
}

inline RAModel& RAModel::operator=(RAModel&& other)
{
  if (this != &other)
  {
    // Clear current model.
    delete raSearch;

    treeType = other.treeType;
    leafSize = other.leafSize;
    randomBasis = other.randomBasis;
    q = std::move(other.q);
    raSearch = std::move(other.raSearch);

    // Reset other model.
    other.treeType = TreeTypes::KD_TREE;
    other.leafSize = 20;
    other.randomBasis = false;
  }

  return *this;
}

// Clean memory, if necessary
inline RAModel::~RAModel()
{
  delete raSearch;
}

inline void RAModel::InitializeModel(const bool naive,
                                     const bool singleMode)
{
  // Clean memory, if necessary.
  delete raSearch;

  switch (treeType)
  {
    case KD_TREE:
      raSearch = new LeafSizeRAWrapper<KDTree>(naive, singleMode);
      break;
    case COVER_TREE:
      raSearch = new RAWrapper<StandardCoverTree>(naive, singleMode);
      break;
    case R_TREE:
      raSearch = new RAWrapper<RTree>(naive, singleMode);
      break;
    case R_STAR_TREE:
      raSearch = new RAWrapper<RStarTree>(naive, singleMode);
      break;
    case X_TREE:
      raSearch = new RAWrapper<XTree>(naive, singleMode);
      break;
    case HILBERT_R_TREE:
      raSearch = new RAWrapper<HilbertRTree>(naive, singleMode);
      break;
    case R_PLUS_TREE:
      raSearch = new RAWrapper<RPlusTree>(naive, singleMode);
      break;
    case R_PLUS_PLUS_TREE:
      raSearch = new RAWrapper<RPlusPlusTree>(naive, singleMode);
      break;
    case UB_TREE:
      raSearch = new LeafSizeRAWrapper<UBTree>(naive, singleMode);
      break;
    case OCTREE:
      raSearch = new LeafSizeRAWrapper<Octree>(naive, singleMode);
      break;
  }
}

inline void RAModel::BuildModel(util::Timers& timers,
                                arma::mat&& referenceSet,
                                const size_t leafSize,
                                const bool naive,
                                const bool singleMode)
{
  // Initialize random basis, if necessary.
  if (randomBasis)
  {
    timers.Start("computing_random_basis");
    Log::Info << "Creating random basis..." << std::endl;
    mlpack::RandomBasis(q, referenceSet.n_rows);

    referenceSet = q * referenceSet;
    timers.Stop("computing_random_basis");
  }

  this->leafSize = leafSize;

  if (!naive)
    Log::Info << "Building reference tree..." << std::endl;

  InitializeModel(naive, singleMode);

  raSearch->Train(timers, std::move(referenceSet), leafSize);

  if (!naive)
    Log::Info << "Tree built." << std::endl;
}

inline void RAModel::Search(util::Timers& timers,
                            arma::mat&& querySet,
                            const size_t k,
                            arma::Mat<size_t>& neighbors,
                            arma::mat& distances)
{
  // Apply the random basis if necessary.
  if (randomBasis)
    querySet = q * querySet;

  Log::Info << "Searching for " << k << " approximate nearest neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree rank-approximate " << TreeName() << " search...";
  else if (!Naive())
    Log::Info << "single-tree rank-approximate " << TreeName() << " search...";
  else
    Log::Info << "brute-force (naive) rank-approximate search...";
  Log::Info << std::endl;

  raSearch->Search(timers, std::move(querySet), k, neighbors, distances,
      leafSize);
}

inline void RAModel::Search(util::Timers& timers,
                            const size_t k,
                            arma::Mat<size_t>& neighbors,
                            arma::mat& distances)
{
  Log::Info << "Searching for " << k << " approximate nearest neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree rank-approximate " << TreeName() << " search...";
  else if (!Naive())
    Log::Info << "single-tree rank-approximate " << TreeName() << " search...";
  else
    Log::Info << "brute-force (naive) rank-approximate search...";
  Log::Info << std::endl;

  raSearch->Search(timers, k, neighbors, distances);
}

inline std::string RAModel::TreeName() const
{
  switch (treeType)
  {
    case KD_TREE:
      return "kd-tree";
    case COVER_TREE:
      return "cover tree";
    case R_TREE:
      return "R tree";
    case R_STAR_TREE:
      return "R* tree";
    case X_TREE:
      return "X tree";
    case HILBERT_R_TREE:
      return "Hilbert R tree";
    case R_PLUS_TREE:
      return "R+ tree";
    case R_PLUS_PLUS_TREE:
      return "R++ tree";
    case UB_TREE:
      return "UB tree";
    case OCTREE:
      return "octree";
    default:
      return "unknown tree";
  }
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RAWrapper<TreeType>::Train(util::Timers& timers,
                                arma::mat&& referenceSet,
                                const size_t /* leafSize */)
{
  if (!ra.Naive())
    timers.Start("tree_building");

  ra.Train(std::move(referenceSet));

  if (!ra.Naive())
    timers.Stop("tree_building");
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RAWrapper<TreeType>::Search(util::Timers& timers,
                                 arma::mat&& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances,
                                 const size_t /* leafSize */)
{
  if (!ra.Naive() && !ra.SingleMode())
  {
    // Build the tree manually, so we can time it.
    timers.Start("tree_building");
    typename decltype(ra)::Tree tree(std::move(querySet));
    timers.Stop("tree_building");

    timers.Start("computing_neighbors");
    ra.Search(&tree, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
  else
  {
    timers.Start("computing_neighbors");
    ra.Search(querySet, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RAWrapper<TreeType>::Search(util::Timers& timers,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  timers.Start("computing_neighbors");
  ra.Search(k, neighbors, distances);
  timers.Stop("computing_neighbors");
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRAWrapper<TreeType>::Train(util::Timers& timers,
                                        arma::mat&& referenceSet,
                                        const size_t leafSize)
{
  // Build tree, if necessary.
  if (ra.Naive())
  {
    ra.Train(std::move(referenceSet));
  }
  else
  {
    timers.Start("tree_building");
    std::vector<size_t> oldFromNewReferences;
    typename decltype(ra)::Tree* tree =
        new typename decltype(ra)::Tree(std::move(referenceSet),
                                        oldFromNewReferences,
                                        leafSize);
    timers.Stop("tree_building");

    ra.Train(tree);

    // Give the model ownership of the tree and the mappings.
    ra.treeOwner = true;
    ra.oldFromNewReferences = std::move(oldFromNewReferences);
  }
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRAWrapper<TreeType>::Search(util::Timers& timers,
                                         arma::mat&& querySet,
                                         const size_t k,
                                         arma::Mat<size_t>& neighbors,
                                         arma::mat& distances,
                                         const size_t leafSize)
{
  if (!ra.Naive() && !ra.SingleMode())
  {
    // Build a second tree and search, taking the leaf size into account.
    timers.Start("tree_building");
    Log::Info << "Building query tree...."<< std::endl;
    std::vector<size_t> oldFromNewQueries;
    typename decltype(ra)::Tree queryTree(std::move(querySet),
                                          oldFromNewQueries,
                                          leafSize);
    Log::Info << "Tree built." << std::endl;
    timers.Stop("tree_building");

    arma::Mat<size_t> neighborsOut;
    arma::mat distancesOut;
    timers.Start("computing_neighbors");
    ra.Search(&queryTree, k, neighborsOut, distancesOut);
    timers.Stop("computing_neighbors");

    // Unmap the query points.
    distances.set_size(distancesOut.n_rows, distancesOut.n_cols);
    neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
    for (size_t i = 0; i < oldFromNewQueries.size(); ++i)
    {
      neighbors.col(oldFromNewQueries[i]) = neighborsOut.col(i);
      distances.col(oldFromNewQueries[i]) = distancesOut.col(i);
    }
  }
  else
  {
    // Search without building a second tree.
    timers.Start("computing_neighbors");
    ra.Search(querySet, k, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

template<typename Archive>
void RAModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(treeType));
  ar(CEREAL_NVP(randomBasis));
  ar(CEREAL_NVP(q));

  // This should never happen, but just in case, be clean with memory.
  if (cereal::is_loading<Archive>())
    InitializeModel(false, false); // Values will be overwritten.

  // Avoid polymorphic serialization by explicitly serializing the correct type.
  switch (treeType)
  {
    case KD_TREE:
      {
        LeafSizeRAWrapper<KDTree>& typedSearch =
            dynamic_cast<LeafSizeRAWrapper<KDTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case COVER_TREE:
      {
        RAWrapper<StandardCoverTree>& typedSearch =
            dynamic_cast<RAWrapper<StandardCoverTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_TREE:
      {
        RAWrapper<RTree>& typedSearch =
            dynamic_cast<RAWrapper<RTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_STAR_TREE:
      {
        RAWrapper<RStarTree>& typedSearch =
            dynamic_cast<RAWrapper<RStarTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case X_TREE:
      {
        RAWrapper<XTree>& typedSearch =
            dynamic_cast<RAWrapper<XTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case HILBERT_R_TREE:
      {
        RAWrapper<HilbertRTree>& typedSearch =
            dynamic_cast<RAWrapper<HilbertRTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_PLUS_TREE:
      {
        RAWrapper<RPlusTree>& typedSearch =
            dynamic_cast<RAWrapper<RPlusTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case R_PLUS_PLUS_TREE:
      {
        RAWrapper<RPlusPlusTree>& typedSearch =
            dynamic_cast<RAWrapper<RPlusPlusTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case UB_TREE:
      {
        LeafSizeRAWrapper<UBTree>& typedSearch =
            dynamic_cast<LeafSizeRAWrapper<UBTree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case OCTREE:
      {
        LeafSizeRAWrapper<Octree>& typedSearch =
            dynamic_cast<LeafSizeRAWrapper<Octree>&>(*raSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
  }
}

} // namespace mlpack

#endif
