/**
 * @file rs_model.cpp
 * @author Ryan Curtin
 *
 * Implementation of the range search model class.
 */
#include "rs_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::range;

/**
 * Initialize the RSModel with the given tree type and whether or not a random
 * basis should be used.
 */
RSModel::RSModel(TreeTypes treeType, bool randomBasis) :
    treeType(treeType),
    randomBasis(randomBasis)
{
  // Nothing to do.
}

// Clean memory, if necessary.
RSModel::~RSModel()
{
  CleanMemory();
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

  // Clean memory, if necessary.
  CleanMemory();

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
      // If necessary, build the tree.
      if (naive)
      {
        rSearch = new RSType<tree::KDTree>(move(referenceSet), naive,
            singleMode);
      }
      else
      {
        vector<size_t> oldFromNewReferences;
        RSType<tree::KDTree>::Tree* kdTree = new RSType<tree::KDTree>::Tree(
            move(referenceSet), oldFromNewReferences, leafSize);
        rSearch = new RSType<tree::KDTree>(kdTree, singleMode);

        // Give the model ownership of the tree and the mappings.
        rSearch->treeOwner = true;
        rSearch->oldFromNewReferences = move(oldFromNewReferences);
      }

      break;

    case COVER_TREE:
      rSearch = new RSType<tree::StandardCoverTree>(move(referenceSet),
          naive, singleMode);
      break;

    case R_TREE:
      rSearch = new RSType<tree::RTree>(move(referenceSet), naive,
          singleMode);
      break;

    case R_STAR_TREE:
      rSearch = new RSType<tree::RStarTree>(move(referenceSet), naive,
          singleMode);
      break;

    case BALL_TREE:
      // If necessary, build the ball tree.
      if (naive)
      {
        rSearch = new RSType<tree::BallTree>(move(referenceSet), naive,
            singleMode);
      }
      else
      {
        vector<size_t> oldFromNewReferences;
        RSType<tree::BallTree>::Tree* ballTree =
            new RSType<tree::BallTree>::Tree(move(referenceSet),
            oldFromNewReferences, leafSize);
        rSearch = new RSType<tree::BallTree>(ballTree, singleMode);

        // Give the model ownership of the tree and the mappings.
        rSearch->treeOwner = true;
        rSearch->oldFromNewReferences = move(oldFromNewReferences);
      }

      break;

    case X_TREE:
      rSearch = new RSType<tree::XTree>(move(referenceSet), naive,
          singleMode);
      break;

    case HILBERT_R_TREE:
      rSearch = new RSType<tree::HilbertRTree>(move(referenceSet), naive,
          singleMode);
      break;

    case R_PLUS_TREE:
      rSearch = new RSType<tree::RPlusTree>(move(referenceSet), naive,
          singleMode);
      break;

    case R_PLUS_PLUS_TREE:
      rSearch = new RSType<tree::RPlusPlusTree>(move(referenceSet),
          naive, singleMode);
      break;

    case VP_TREE:
      rSearch = new RSType<tree::VPTree>(move(referenceSet), naive,
          singleMode);
      break;

    case RP_TREE:
      rSearch = new RSType<tree::RPTree>(move(referenceSet), naive,
          singleMode);
      break;

    case MAX_RP_TREE:
      rSearch = new RSType<tree::MaxRPTree>(move(referenceSet),
          naive, singleMode);
      break;

    case UB_TREE:
      rSearch = new RSType<tree::UBTree>(move(referenceSet),
          naive, singleMode);
      break;
  }

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
    
  BiSearchVistor search(querySet, range, neighbors, distances, leafSize);
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
    default:
      return "unknown tree";
  }
}

// Clean memory.
void RSModel::CleanMemory()
{
    boost::apply_visitor(DeleteVisitor(), rSearch);
}
