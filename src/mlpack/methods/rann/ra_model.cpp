/**
 * @file methods/rann/ra_model.cpp
 * @author Ryan Curtin
 *
 * Implementation of the RAModel class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "ra_model.hpp"
#include <mlpack/core/math/random_basis.hpp>

namespace mlpack {
namespace neighbor {

RAModel::RAModel(const TreeTypes treeType, const bool randomBasis) :
    treeType(treeType),
    leafSize(20),
    randomBasis(randomBasis),
    raSearch(NULL)
{
  // Nothing to do.
}

// Copy constructor.
RAModel::RAModel(const RAModel& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    q(other.q),
    raSearch(other.raSearch->Clone())
{
  // Nothing to do.
}

// Move constructor.
RAModel::RAModel(RAModel&& other) :
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
RAModel& RAModel::operator=(const RAModel& other)
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

RAModel& RAModel::operator=(RAModel&& other)
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
RAModel::~RAModel()
{
  delete raSearch;
}

void RAModel::InitializeModel(const bool naive, const bool singleMode)
{
  // Clean memory, if necessary.
  delete raSearch;

  switch (treeType)
  {
    case KD_TREE:
      raSearch = new LeafSizeRAWrapper<tree::KDTree>(naive, singleMode);
      break;
    case COVER_TREE:
      raSearch = new RAWrapper<tree::StandardCoverTree>(naive, singleMode);
      break;
    case R_TREE:
      raSearch = new RAWrapper<tree::RTree>(naive, singleMode);
      break;
    case R_STAR_TREE:
      raSearch = new RAWrapper<tree::RStarTree>(naive, singleMode);
      break;
    case X_TREE:
      raSearch = new RAWrapper<tree::XTree>(naive, singleMode);
      break;
    case HILBERT_R_TREE:
      raSearch = new RAWrapper<tree::HilbertRTree>(naive, singleMode);
      break;
    case R_PLUS_TREE:
      raSearch = new RAWrapper<tree::RPlusTree>(naive, singleMode);
      break;
    case R_PLUS_PLUS_TREE:
      raSearch = new RAWrapper<tree::RPlusPlusTree>(naive, singleMode);
      break;
    case UB_TREE:
      raSearch = new RAWrapper<tree::UBTree>(naive, singleMode);
      break;
    case OCTREE:
      raSearch = new LeafSizeRAWrapper<tree::Octree>(naive, singleMode);
      break;
  }
}

void RAModel::BuildModel(arma::mat&& referenceSet,
                         const size_t leafSize,
                         const bool naive,
                         const bool singleMode)
{
  // Initialize random basis, if necessary.
  if (randomBasis)
  {
    Log::Info << "Creating random basis..." << std::endl;
    math::RandomBasis(q, referenceSet.n_rows);
  }

  this->leafSize = leafSize;

  if (randomBasis)
    referenceSet = q * referenceSet;

  if (!naive)
  {
    Timer::Start("tree_building");
    Log::Info << "Building reference tree..." << std::endl;
  }

  InitializeModel(naive, singleMode);

  raSearch->Train(std::move(referenceSet), leafSize);

  if (!naive)
  {
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << std::endl;
  }
}

void RAModel::Search(arma::mat&& querySet,
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

  raSearch->Search(std::move(querySet), k, neighbors, distances, leafSize);
}

void RAModel::Search(const size_t k,
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

  raSearch->Search(k, neighbors, distances);
}

std::string RAModel::TreeName() const
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

} // namespace neighbor
} // namespace mlpack
