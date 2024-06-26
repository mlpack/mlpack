/**
 * @file methods/range_search/rs_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of serialize() and inline functions for RSModel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_IMPL_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "rs_model.hpp"

#include <mlpack/core/math/random_basis.hpp>

namespace mlpack {

/**
 * Initialize the RSModel with the given tree type and whether or not a random
 * basis should be used.
 */
inline RSModel::RSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    leafSize(0),
    randomBasis(randomBasis),
    rSearch(NULL)
{
  // Nothing to do.
}

// Copy constructor.
inline RSModel::RSModel(const RSModel& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(other.q),
    rSearch(other.rSearch->Clone())
{
  // Nothing to do.
}

// Move constructor.
inline RSModel::RSModel(RSModel&& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(std::move(other.q)),
    rSearch(std::move(other.rSearch))
{
  // Reset other model.
  other.treeType = TreeTypes::KD_TREE;
  other.leafSize = 0;
  other.randomBasis = false;
}

// Copy operator.
inline RSModel& RSModel::operator=(const RSModel& other)
{
  if (this != &other)
  {
    delete rSearch;

    treeType = other.treeType;
    leafSize = other.leafSize;
    randomBasis = other.randomBasis;
    q = other.q;
    rSearch = other.rSearch->Clone();
  }

  return *this;
}

// Move operator.
inline RSModel& RSModel::operator=(RSModel&& other)
{
  if (this != &other)
  {
    delete rSearch;

    treeType = other.treeType;
    leafSize = other.leafSize;
    randomBasis = other.randomBasis;
    q = std::move(other.q);
    rSearch = std::move(other.rSearch);

    other.treeType = TreeTypes::KD_TREE;
    other.leafSize = 0;
    other.randomBasis = false;
  }

  return *this;
}

// Clean memory, if necessary.
inline RSModel::~RSModel()
{
  delete rSearch;
}

inline void RSModel::InitializeModel(const bool naive,
                                     const bool singleMode)
{
  // Clean memory, if necessary.
  delete rSearch;

  switch (treeType)
  {
    case KD_TREE:
      rSearch = new LeafSizeRSWrapper<KDTree>(naive, singleMode);
      break;

    case COVER_TREE:
      rSearch = new RSWrapper<StandardCoverTree>(naive, singleMode);
      break;

    case R_TREE:
      rSearch = new RSWrapper<RTree>(naive, singleMode);
      break;

    case R_STAR_TREE:
      rSearch = new RSWrapper<RStarTree>(naive, singleMode);
      break;

    case BALL_TREE:
      rSearch = new LeafSizeRSWrapper<BallTree>(naive, singleMode);
      break;

    case X_TREE:
      rSearch = new RSWrapper<XTree>(naive, singleMode);
      break;

    case HILBERT_R_TREE:
      rSearch = new RSWrapper<HilbertRTree>(naive, singleMode);
      break;

    case R_PLUS_TREE:
      rSearch = new RSWrapper<RPlusTree>(naive, singleMode);
      break;

    case R_PLUS_PLUS_TREE:
      rSearch = new RSWrapper<RPlusPlusTree>(naive, singleMode);
      break;

    case VP_TREE:
      rSearch = new LeafSizeRSWrapper<VPTree>(naive, singleMode);
      break;

    case RP_TREE:
      rSearch = new LeafSizeRSWrapper<RPTree>(naive, singleMode);
      break;

    case MAX_RP_TREE:
      rSearch = new LeafSizeRSWrapper<MaxRPTree>(naive, singleMode);
      break;

    case UB_TREE:
      rSearch = new LeafSizeRSWrapper<UBTree>(naive, singleMode);
      break;

    case OCTREE:
      rSearch = new LeafSizeRSWrapper<Octree>(naive, singleMode);
      break;
  }
}

inline void RSModel::BuildModel(util::Timers& timers,
                                arma::mat&& referenceSet,
                                const size_t leafSize,
                                const bool naive,
                                const bool singleMode)
{
  // Initialize random basis if necessary.
  if (randomBasis)
  {
    timers.Start("computing_random_basis");
    Log::Info << "Creating random basis..." << std::endl;
    mlpack::RandomBasis(q, referenceSet.n_rows);

    // Do we need to modify the reference set?
    if (randomBasis)
      referenceSet = q * referenceSet;
    timers.Stop("computing_random_basis");
  }

  this->leafSize = leafSize;

  if (!naive)
    Log::Info << "Building reference tree..." << std::endl;

  InitializeModel(naive, singleMode);

  rSearch->Train(timers, std::move(referenceSet), leafSize);

  if (!naive)
    Log::Info << "Tree built." << std::endl;
}

// Perform range search.
inline void RSModel::Search(util::Timers& timers,
                            arma::mat&& querySet,
                            const Range& range,
                            std::vector<std::vector<size_t>>& neighbors,
                            std::vector<std::vector<double>>& distances)
{
  // We may need to map the query set randomly.
  if (randomBasis)
  {
    timers.Start("applying_random_basis");
    querySet = q * querySet;
    timers.Stop("applying_random_basis");
  }

  Log::Info << "Search for points in the range [" << range.Lo() << ", "
      << range.Hi() << "] with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree " << TreeName() << " search..." << std::endl;
  else if (!Naive())
    Log::Info << "single-tree " << TreeName() << " search..." << std::endl;
  else
    Log::Info << "brute-force (naive) search..." << std::endl;

  rSearch->Search(timers, std::move(querySet), range, neighbors, distances,
      leafSize);
}

// Perform range search (monochromatic case).
inline void RSModel::Search(util::Timers& timers,
                            const Range& range,
                            std::vector<std::vector<size_t>>& neighbors,
                            std::vector<std::vector<double>>& distances)
{
  Log::Info << "Search for points in the range [" << range.Lo() << ", "
      << range.Hi() << "] with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree " << TreeName() << " search..." << std::endl;
  else if (!Naive())
    Log::Info << "single-tree " << TreeName() << " search..." << std::endl;
  else
    Log::Info << "brute-force (naive) search..." << std::endl;

  rSearch->Search(timers, range, neighbors, distances);
}

// Get the name of the tree type.
inline std::string RSModel::TreeName() const
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
    case BALL_TREE:
      return "ball tree";
    case X_TREE:
      return "X tree";
    case HILBERT_R_TREE:
      return "Hilbert R tree";
    case R_PLUS_TREE:
      return "R+ tree";
    case R_PLUS_PLUS_TREE:
      return "R++ tree";
    case VP_TREE:
      return "vantage point tree";
    case RP_TREE:
      return "random projection tree (mean split)";
    case MAX_RP_TREE:
      return "random projection tree (max split)";
    case UB_TREE:
      return "UB tree";
    case OCTREE:
      return "octree";
    default:
      return "unknown tree";
  }
}

// Clean memory.
inline void RSModel::CleanMemory()
{
  delete rSearch;
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RSWrapper<TreeType>::Train(util::Timers& timers,
                                arma::mat&& referenceSet,
                                const size_t /* leafSize */)
{
  if (!Naive())
    timers.Start("tree_building");

  rs.Train(std::move(referenceSet));
  if (!Naive())
    timers.Stop("tree_building");
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RSWrapper<TreeType>::Search(util::Timers& timers,
                                 arma::mat&& querySet,
                                 const Range& range,
                                 std::vector<std::vector<size_t>>& neighbors,
                                 std::vector<std::vector<double>>& distances,
                                 const size_t /* leafSize */)
{
  if (!Naive() && !SingleMode())
  {
    // We build the query tree manually, so that we can time how long it takes.
    timers.Start("tree_building");
    typename decltype(rs)::Tree queryTree(std::move(querySet));
    timers.Stop("tree_building");

    timers.Start("computing_neighbors");
    rs.Search(&queryTree, range, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
  else
  {
    timers.Start("computing_neighbors");
    rs.Search(std::move(querySet), range, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void RSWrapper<TreeType>::Search(util::Timers& timers,
                                 const Range& range,
                                 std::vector<std::vector<size_t>>& neighbors,
                                 std::vector<std::vector<double>>& distances)
{
  timers.Start("computing_neighbors");
  rs.Search(range, neighbors, distances);
  timers.Stop("computing_neighbors");
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRSWrapper<TreeType>::Train(util::Timers& timers,
                                        arma::mat&& referenceSet,
                                        const size_t leafSize)
{
  if (rs.Naive())
  {
    rs.Train(std::move(referenceSet));
  }
  else
  {
    timers.Start("tree_building");
    std::vector<size_t> oldFromNewReferences;
    typename decltype(rs)::Tree* tree =
        new typename decltype(rs)::Tree(std::move(referenceSet),
                                        oldFromNewReferences,
                                        leafSize);
    rs.Train(tree);

    // Give the model ownership of the tree and the mappings.
    rs.treeOwner = true;
    rs.oldFromNewReferences = std::move(oldFromNewReferences);
    timers.Stop("tree_building");
  }
}

template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void LeafSizeRSWrapper<TreeType>::Search(
    util::Timers& timers,
    arma::mat&& querySet,
    const Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances,
    const size_t leafSize)
{
  if (!rs.Naive() && !rs.SingleMode())
  {
    // Build a second tree and search.
    timers.Start("tree_building");
    Log::Info << "Building query tree..." << std::endl;
    std::vector<size_t> oldFromNewQueries;
    typename decltype(rs)::Tree queryTree(std::move(querySet),
                                          oldFromNewQueries,
                                          leafSize);
    Log::Info << "Tree built." << std::endl;
    timers.Stop("tree_building");

    std::vector<std::vector<size_t>> neighborsOut;
    std::vector<std::vector<double>> distancesOut;
    timers.Start("computing_neighbors");
    rs.Search(&queryTree, range, neighborsOut, distancesOut);
    timers.Stop("computing_neighbors");

    // Remap the query points.
    neighbors.resize(queryTree.Dataset().n_cols);
    distances.resize(queryTree.Dataset().n_cols);
    for (size_t i = 0; i < queryTree.Dataset().n_cols; ++i)
    {
      neighbors[oldFromNewQueries[i]] = neighborsOut[i];
      distances[oldFromNewQueries[i]] = distancesOut[i];
    }
  }
  else
  {
    timers.Start("computing_neighbors");
    rs.Search(std::move(querySet), range, neighbors, distances);
    timers.Stop("computing_neighbors");
  }
}

// Serialize the model.
template<typename Archive>
void RSModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(treeType));
  ar(CEREAL_NVP(randomBasis));
  ar(CEREAL_NVP(q));

  // This should never happen, but just in case...
  if (cereal::is_loading<Archive>())
    InitializeModel(false, false); // Values will be overwritten.

  // Avoid polymorphic serialization by explicitly serializing the correct type.
  switch (treeType)
  {
    case KD_TREE:
      {
        LeafSizeRSWrapper<KDTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<KDTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case COVER_TREE:
      {
        RSWrapper<StandardCoverTree>& typedSearch =
            dynamic_cast<RSWrapper<StandardCoverTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_TREE:
      {
        RSWrapper<RTree>& typedSearch =
            dynamic_cast<RSWrapper<RTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_STAR_TREE:
      {
        RSWrapper<RStarTree>& typedSearch =
            dynamic_cast<RSWrapper<RStarTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case BALL_TREE:
      {
        LeafSizeRSWrapper<BallTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<BallTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case X_TREE:
      {
        RSWrapper<XTree>& typedSearch =
            dynamic_cast<RSWrapper<XTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case HILBERT_R_TREE:
      {
        RSWrapper<HilbertRTree>& typedSearch =
            dynamic_cast<RSWrapper<HilbertRTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_PLUS_TREE:
      {
        RSWrapper<RPlusTree>& typedSearch =
            dynamic_cast<RSWrapper<RPlusTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case R_PLUS_PLUS_TREE:
      {
        RSWrapper<RPlusPlusTree>& typedSearch =
            dynamic_cast<RSWrapper<RPlusPlusTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case VP_TREE:
      {
        LeafSizeRSWrapper<VPTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<VPTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case RP_TREE:
      {
        LeafSizeRSWrapper<RPTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<RPTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }

    case MAX_RP_TREE:
      {
        LeafSizeRSWrapper<MaxRPTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<MaxRPTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case UB_TREE:
      {
        LeafSizeRSWrapper<UBTree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<UBTree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
    case OCTREE:
      {
        LeafSizeRSWrapper<Octree>& typedSearch =
            dynamic_cast<LeafSizeRSWrapper<Octree>&>(*rSearch);
        ar(CEREAL_NVP(typedSearch));
        break;
      }
  }
}

} // namespace mlpack

#endif
