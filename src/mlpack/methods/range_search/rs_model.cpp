/**
 * @file rs_model.cpp
 * @author Ryan Curtin
 *
 * Implementation of the range search model class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "rs_model.hpp"
#include <mlpack/core/math/random_basis.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::range;

/**
 * Initialize the RSModel with the given tree type and whether or not a random
 * basis should be used.
 */
RSModel::RSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    leafSize(0),
    randomBasis(randomBasis)
{
  // Nothing to do.
}

// Copy constructor.
RSModel::RSModel(const RSModel& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    rSearch(other.rSearch)
{

}

// Move constructor.
RSModel::RSModel(RSModel&& other) :
    treeType(other.treeType),
    leafSize(other.leafSize),
    randomBasis(other.randomBasis),
    rSearch(other.rSearch)
{
  // Reset other model.
  other.treeType = TreeTypes::KD_TREE;
  other.leafSize = 0;
  other.randomBasis = false;
  other.rSearch = decltype(other.rSearch)();
}

// Copy operator.
RSModel& RSModel::operator=(const RSModel& other)
{
  boost::apply_visitor(DeleteVisitor(), rSearch);

  treeType = other.treeType;
  leafSize = other.leafSize;
  randomBasis = other.randomBasis;
  rSearch = other.rSearch;

  return *this;
}

// Move operator.
RSModel& RSModel::operator=(RSModel&& other)
{
  boost::apply_visitor(DeleteVisitor(), rSearch);

  treeType = other.treeType;
  leafSize = other.leafSize;
  randomBasis = other.randomBasis;
  rSearch = other.rSearch;

  // Reset other model.
  other.treeType = TreeTypes::KD_TREE;
  other.leafSize = 0;
  other.randomBasis = false;
  other.rSearch = decltype(other.rSearch)();

  return *this;
}

// Clean memory, if necessary.
RSModel::~RSModel()
{
  boost::apply_visitor(DeleteVisitor(), rSearch);
}

void RSModel::BuildModel(arma::mat&& referenceSet,
                         const size_t leafSize,
                         const bool naive,
                         const bool singleMode)
{
  // Initialize random basis if necessary.
  if (randomBasis)
  {
    Log::Info << "Creating random basis..." << endl;
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
    Log::Info << "Building reference tree..." << endl;
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
      rSearch = new RSType<tree::RTree>(naive,singleMode);
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
    Log::Info << "Tree built." << endl;
  }
}

// Perform range search.
void RSModel::Search(arma::mat&& querySet,
                     const math::Range& range,
                     vector<vector<size_t>>& neighbors,
                     vector<vector<double>>& distances)
{
  // We may need to map the query set randomly.
  if (randomBasis)
    querySet = q * querySet;

  Log::Info << "Search for points in the range [" << range.Lo() << ", "
      << range.Hi() << "] with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree " << TreeName() << " search..." << endl;
  else if (!Naive())
    Log::Info << "single-tree " << TreeName() << " search..." << endl;
  else
    Log::Info << "brute-force (naive) search..." << endl;


  BiSearchVisitor search(querySet, range, neighbors, distances,
      leafSize);
  boost::apply_visitor(search, rSearch);
}

// Perform range search (monochromatic case).
void RSModel::Search(const math::Range& range,
                     vector<vector<size_t>>& neighbors,
                     vector<vector<double>>& distances)
{
  Log::Info << "Search for points in the range [" << range.Lo() << ", "
      << range.Hi() << "] with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree " << TreeName() << " search..." << endl;
  else if (!Naive())
    Log::Info << "single-tree " << TreeName() << " search..." << endl;
  else
    Log::Info << "brute-force (naive) search..." << endl;

  MonoSearchVisitor search(range, neighbors, distances);
  boost::apply_visitor(search, rSearch);
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
  boost::apply_visitor(DeleteVisitor(), rSearch);
}
