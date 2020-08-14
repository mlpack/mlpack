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
namespace range {

/**
 * Initialize the RSModel with the given tree type and whether or not a random
 * basis should be used.
 */
inline RSModel::RSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    leafSize(0),
    randomBasis(randomBasis)
{
  // Nothing to do.
}

// Copy constructor.
inline RSModel::RSModel(const RSModel& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(other.q),
    rSearch(other.rSearch)
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
  other.rSearch = decltype(other.rSearch)();
}

inline RSModel& RSModel::operator=(RSModel other)
{
  boost::apply_visitor(DeleteVisitor(), rSearch);

  treeType = other.treeType;
  leafSize = other.leafSize;
  randomBasis = other.randomBasis;
  q = std::move(other.q);
  rSearch = std::move(other.rSearch);

  return *this;
}

// Clean memory, if necessary.
inline RSModel::~RSModel()
{
  boost::apply_visitor(DeleteVisitor(), rSearch);
}

inline void RSModel::BuildModel(arma::mat&& referenceSet,
                                const size_t leafSize,
                                const bool naive,
                                const bool singleMode)
{
  // Initialize random basis if necessary.
  if (randomBasis)
  {
    Log::Info << "Creating random basis..." << std::endl;
    math::RandomBasis(q, referenceSet.n_rows);
  }

  this->leafSize = leafSize;

  // Clean memory, if necessary.
  boost::apply_visitor(DeleteVisitor(), rSearch);

  // Do we need to modify the reference set?
  if (randomBasis)
    referenceSet = q * referenceSet;

  if (!naive)
  {
    Timer::Start("tree_building");
    Log::Info << "Building reference tree..." << std::endl;
  }

  switch (treeType)
  {
    case KD_TREE:
      rSearch = new RSType<tree::KDTree> (naive, singleMode);
      break;

    case COVER_TREE:
      rSearch = new RSType<tree::StandardCoverTree>(naive, singleMode);
      break;

    case R_TREE:
      rSearch = new RSType<tree::RTree>(naive, singleMode);
      break;

    case R_STAR_TREE:
      rSearch = new RSType<tree::RStarTree>(naive, singleMode);
      break;

    case BALL_TREE:
      rSearch = new RSType<tree::BallTree>(naive, singleMode);
      break;

    case X_TREE:
      rSearch = new RSType<tree::XTree>(naive, singleMode);
      break;

    case HILBERT_R_TREE:
      rSearch = new RSType<tree::HilbertRTree>(naive, singleMode);
      break;

    case R_PLUS_TREE:
      rSearch = new RSType<tree::RPlusTree>(naive, singleMode);
      break;

    case R_PLUS_PLUS_TREE:
      rSearch = new RSType<tree::RPlusPlusTree>(naive, singleMode);
      break;

    case VP_TREE:
      rSearch = new RSType<tree::VPTree>(naive, singleMode);
      break;

    case RP_TREE:
      rSearch = new RSType<tree::RPTree>(naive, singleMode);
      break;

    case MAX_RP_TREE:
      rSearch = new RSType<tree::MaxRPTree>(naive, singleMode);
      break;

    case UB_TREE:
      rSearch = new RSType<tree::UBTree>(naive, singleMode);
      break;

    case OCTREE:
      rSearch = new RSType<tree::Octree>(naive, singleMode);
      break;
  }

  TrainVisitor tn(std::move(referenceSet), leafSize);
  boost::apply_visitor(tn, rSearch);

  if (!naive)
  {
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << std::endl;
  }
}

// Perform range search.
inline void RSModel::Search(arma::mat&& querySet,
                            const math::Range& range,
                            std::vector<std::vector<size_t>>& neighbors,
                            std::vector<std::vector<double>>& distances)
{
  // We may need to map the query set randomly.
  if (randomBasis)
    querySet = q * querySet;

  Log::Info << "Search for points in the range [" << range.Lo() << ", "
      << range.Hi() << "] with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree " << TreeName() << " search..." << std::endl;
  else if (!Naive())
    Log::Info << "single-tree " << TreeName() << " search..." << std::endl;
  else
    Log::Info << "brute-force (naive) search..." << std::endl;


  BiSearchVisitor search(querySet, range, neighbors, distances,
      leafSize);
  boost::apply_visitor(search, rSearch);
}

// Perform range search (monochromatic case).
inline void RSModel::Search(const math::Range& range,
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

  MonoSearchVisitor search(range, neighbors, distances);
  boost::apply_visitor(search, rSearch);
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
  boost::apply_visitor(DeleteVisitor(), rSearch);
}

//! Monochromatic range search on the given RSType instance.
template<typename RSType>
void MonoSearchVisitor::operator()(RSType* rs) const
{
  if (rs)
    return rs->Search(range, neighbors, distances);
  throw std::runtime_error("no range search model initialized");
}

//! Save parameters for bichromatic range search.
inline BiSearchVisitor::BiSearchVisitor(
    const arma::mat& querySet,
    const math::Range& range,
    std::vector<std::vector<size_t>>& neighbors,
    std::vector<std::vector<double>>& distances,
    const size_t leafSize) :
    querySet(querySet),
    range(range),
    neighbors(neighbors),
    distances(distances),
    leafSize(leafSize)
{}

//! Default Bichromatic range search on the given RSType instance.
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void BiSearchVisitor::operator()(RSTypeT<TreeType>* rs) const
{
  if (rs)
    return rs->Search(querySet, range, neighbors, distances);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search on the given RSType specialized for KDTrees.
inline void BiSearchVisitor::operator()(RSTypeT<tree::KDTree>* rs) const
{
  if (rs)
    return SearchLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search on the given RSType specialized for BallTrees.
inline void BiSearchVisitor::operator()(RSTypeT<tree::BallTree>* rs) const
{
  if (rs)
    return SearchLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search specialized for Ocrees.
inline void BiSearchVisitor::operator()(RSTypeT<tree::Octree>* rs) const
{
  if (rs)
    return SearchLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Bichromatic range search on the given RSType considering the leafSize.
template<typename RSType>
void BiSearchVisitor::SearchLeaf(RSType* rs) const
{
  if (!rs->Naive() && !rs->SingleMode())
  {
    // Build a second tree and search.
    Timer::Start("tree_building");
    Log::Info << "Building query tree..." << std::endl;
    std::vector<size_t> oldFromNewQueries;
    typename RSType::Tree queryTree(std::move(querySet), oldFromNewQueries,
        leafSize);
    Log::Info << "Tree built." << std::endl;
    Timer::Stop("tree_building");

    std::vector<std::vector<size_t>> neighborsOut;
    std::vector<std::vector<double>> distancesOut;
    rs->Search(&queryTree, range, neighborsOut, distancesOut);

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
    rs->Search(querySet, range, neighbors, distances);
}

//! Save parameters for Train.
inline TrainVisitor::TrainVisitor(arma::mat&& referenceSet,
                                  const size_t leafSize) :
    referenceSet(std::move(referenceSet)),
    leafSize(leafSize)
{}

//! Default Train on the given RSType instance.
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void TrainVisitor::operator()(RSTypeT<TreeType>* rs) const
{
  if (rs)
    return rs->Train(std::move(referenceSet));
  throw std::runtime_error("no range search model initialized");
}

//! Train on the given RSType specialized for KDTrees.
inline void TrainVisitor::operator()(RSTypeT<tree::KDTree>* rs) const
{
  if (rs)
    return TrainLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Train on the given RSType specialized for BallTrees.
inline void TrainVisitor::operator()(RSTypeT<tree::BallTree>* rs) const
{
  if (rs)
    return TrainLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Train specialized for Octrees.
inline void TrainVisitor::operator()(RSTypeT<tree::Octree>* rs) const
{
  if (rs)
    return TrainLeaf(rs);
  throw std::runtime_error("no range search model initialized");
}

//! Train on the given RSType considering the leafSize.
template<typename RSType>
void TrainVisitor::TrainLeaf(RSType* rs) const
{
  if (rs->Naive())
    rs->Train(std::move(referenceSet));
  else
  {
    std::vector<size_t> oldFromNewReferences;
    typename RSType::Tree* tree =
        new typename RSType::Tree(std::move(referenceSet), oldFromNewReferences,
        leafSize);
    rs->Train(tree);

    // Give the model ownership of the tree and the mappings.
    rs->treeOwner = true;
    rs->oldFromNewReferences = std::move(oldFromNewReferences);
  }
}

//! Expose the referenceSet of the given RSType.
template<typename RSType>
const arma::mat& ReferenceSetVisitor::operator()(RSType* rs) const
{
  if (rs)
    return rs->ReferenceSet();
  throw std::runtime_error("no range search model initialized");
}

//! For cleaning memory
template<typename RSType>
void DeleteVisitor::operator()(RSType* rs) const
{
  if (rs)
    delete rs;
}

//! Return whether single mode enabled
template<typename RSType>
bool& SingleModeVisitor::operator()(RSType* rs) const
{
  if (rs)
    return rs->SingleMode();
  throw std::runtime_error("no range search model initialized");
}

//! Exposes Naive() function of given RSType
template<typename RSType>
bool& NaiveVisitor::operator()(RSType* rs) const
{
  if (rs)
    return rs->Naive();
  throw std::runtime_error("no range search model initialized");
}

// Serialize the model.
template<typename Archive>
void RSModel::serialize(Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar(CEREAL_NVP(treeType));
  ar(CEREAL_NVP(randomBasis));
  ar(CEREAL_NVP(q));

  // This should never happen, but just in case...
  if (Archive::is_loading::value)
    boost::apply_visitor(DeleteVisitor(), rSearch);

  // We'll only need to serialize one of the model objects, based on the type.
  ar(CEREAL_VARIANT_POINTER(rSearch));
}

inline const arma::mat& RSModel::Dataset() const
{
  return boost::apply_visitor(ReferenceSetVisitor(), rSearch);
}

inline bool RSModel::SingleMode() const
{
  return boost::apply_visitor(SingleModeVisitor(), rSearch);
}

inline bool& RSModel::SingleMode()
{
  return boost::apply_visitor(SingleModeVisitor(), rSearch);
}

inline bool RSModel::Naive() const
{
  return boost::apply_visitor(NaiveVisitor(), rSearch);
}

inline bool& RSModel::Naive()
{
  return boost::apply_visitor(NaiveVisitor(), rSearch);
}

} // namespace range
} // namespace mlpack

#endif
