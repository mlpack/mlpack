/**
 * @file methods/range_search/rs_model.cpp
 * @author Ryan Curtin
 *
 * Implementation of serialize() and inline functions for RSModel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "rs_model.hpp"

#include <mlpack/core/math/random_basis.hpp>

namespace mlpack {
namespace range {

/**
 * Initialize the RSModel with the given tree type and whether or not a random
 * basis should be used.
 */
RSModel::RSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    leafSize(0),
    randomBasis(randomBasis),
    rSearch(NULL)
{
  // Nothing to do.
}

// Copy constructor.
RSModel::RSModel(const RSModel& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(other.q),
    rSearch(other.rSearch->Clone())
{
  // Nothing to do.
}

// Move constructor.
RSModel::RSModel(RSModel&& other) :
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
RSModel& RSModel::operator=(const RSModel& other)
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
RSModel& RSModel::operator=(RSModel&& other)
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
RSModel::~RSModel()
{
  delete rSearch;
}

void RSModel::InitializeModel(const bool naive, const bool singleMode)
{
  // Clean memory, if necessary.
  delete rSearch;

  switch (treeType)
  {
    case KD_TREE:
      rSearch = new LeafSizeRSWrapper<tree::KDTree>(naive, singleMode);
      break;

    case COVER_TREE:
      rSearch = new RSWrapper<tree::StandardCoverTree>(naive, singleMode);
      break;

    case R_TREE:
      rSearch = new RSWrapper<tree::RTree>(naive, singleMode);
      break;

    case R_STAR_TREE:
      rSearch = new RSWrapper<tree::RStarTree>(naive, singleMode);
      break;

    case BALL_TREE:
      rSearch = new LeafSizeRSWrapper<tree::BallTree>(naive, singleMode);
      break;

    case X_TREE:
      rSearch = new RSWrapper<tree::XTree>(naive, singleMode);
      break;

    case HILBERT_R_TREE:
      rSearch = new RSWrapper<tree::HilbertRTree>(naive, singleMode);
      break;

    case R_PLUS_TREE:
      rSearch = new RSWrapper<tree::RPlusTree>(naive, singleMode);
      break;

    case R_PLUS_PLUS_TREE:
      rSearch = new RSWrapper<tree::RPlusPlusTree>(naive, singleMode);
      break;

    case VP_TREE:
      rSearch = new RSWrapper<tree::VPTree>(naive, singleMode);
      break;

    case RP_TREE:
      rSearch = new RSWrapper<tree::RPTree>(naive, singleMode);
      break;

    case MAX_RP_TREE:
      rSearch = new RSWrapper<tree::MaxRPTree>(naive, singleMode);
      break;

    case UB_TREE:
      rSearch = new RSWrapper<tree::UBTree>(naive, singleMode);
      break;

    case OCTREE:
      rSearch = new LeafSizeRSWrapper<tree::Octree>(naive, singleMode);
      break;
  }
}

void RSModel::BuildModel(arma::mat&& referenceSet,
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

  // Do we need to modify the reference set?
  if (randomBasis)
    referenceSet = q * referenceSet;

  if (!naive)
  {
    Timer::Start("tree_building");
    Log::Info << "Building reference tree..." << std::endl;
  }

  InitializeModel(naive, singleMode);

  rSearch->Train(std::move(referenceSet), leafSize);

  if (!naive)
  {
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << std::endl;
  }
}

// Perform range search.
void RSModel::Search(arma::mat&& querySet,
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

  rSearch->Search(std::move(querySet), range, neighbors, distances, leafSize);
}

// Perform range search (monochromatic case).
void RSModel::Search(const math::Range& range,
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

  rSearch->Search(range, neighbors, distances);
}

// Get the name of the tree type.
std::string RSModel::TreeName() const
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
void RSModel::CleanMemory()
{
  delete rSearch;
}

} // namespace range
} // namespace mlpack
