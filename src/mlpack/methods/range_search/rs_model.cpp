/**
 * @file rs_model.cpp
 * @author Ryan Curtin
 *
 * Implementation of the range search model class.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
    randomBasis(randomBasis),
    kdTreeRS(NULL),
    coverTreeRS(NULL),
    rTreeRS(NULL),
    rStarTreeRS(NULL),
    ballTreeRS(NULL),
    xTreeRS(NULL)
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
        kdTreeRS = new RSType<tree::KDTree>(move(referenceSet), naive,
            singleMode);
      }
      else
      {
        vector<size_t> oldFromNewReferences;
        RSType<tree::KDTree>::Tree* kdTree = new RSType<tree::KDTree>::Tree(
            move(referenceSet), oldFromNewReferences, leafSize);
        kdTreeRS = new RSType<tree::KDTree>(kdTree, singleMode);

        // Give the model ownership of the tree and the mappings.
        kdTreeRS->treeOwner = true;
        kdTreeRS->oldFromNewReferences = move(oldFromNewReferences);
      }

      break;

    case COVER_TREE:
      coverTreeRS = new RSType<tree::StandardCoverTree>(move(referenceSet),
          naive, singleMode);
      break;

    case R_TREE:
      rTreeRS = new RSType<tree::RTree>(move(referenceSet), naive,
          singleMode);
      break;

    case R_STAR_TREE:
      rStarTreeRS = new RSType<tree::RStarTree>(move(referenceSet), naive,
          singleMode);
      break;

    case BALL_TREE:
      // If necessary, build the ball tree.
      if (naive)
      {
        ballTreeRS = new RSType<tree::BallTree>(move(referenceSet), naive,
            singleMode);
      }
      else
      {
        vector<size_t> oldFromNewReferences;
        RSType<tree::BallTree>::Tree* ballTree =
            new RSType<tree::BallTree>::Tree(move(referenceSet),
            oldFromNewReferences, leafSize);
        ballTreeRS = new RSType<tree::BallTree>(ballTree, singleMode);

        // Give the model ownership of the tree and the mappings.
        ballTreeRS->treeOwner = true;
        ballTreeRS->oldFromNewReferences = move(oldFromNewReferences);
      }

      break;

    case X_TREE:
      xTreeRS = new RSType<tree::XTree>(move(referenceSet), naive,
          singleMode);
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

  switch (treeType)
  {
    case KD_TREE:
      if (!kdTreeRS->Naive() && !kdTreeRS->SingleMode())
      {
        // Build a second tree and search.
        Timer::Start("tree_building");
        Log::Info << "Building query tree..." << endl;
        vector<size_t> oldFromNewQueries;
        RSType<tree::KDTree>::Tree queryTree(move(querySet), oldFromNewQueries,
            leafSize);
        Log::Info << "Tree built." << endl;
        Timer::Stop("tree_building");

        vector<vector<size_t>> neighborsOut;
        vector<vector<double>> distancesOut;
        kdTreeRS->Search(&queryTree, range, neighborsOut, distancesOut);

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
        // Search without building a second tree.
        kdTreeRS->Search(querySet, range, neighbors, distances);
      }
      break;

    case COVER_TREE:
      coverTreeRS->Search(querySet, range, neighbors, distances);
      break;

    case R_TREE:
      rTreeRS->Search(querySet, range, neighbors, distances);
      break;

    case R_STAR_TREE:
      rStarTreeRS->Search(querySet, range, neighbors, distances);
      break;

    case BALL_TREE:
      if (!ballTreeRS->Naive() && !ballTreeRS->SingleMode())
      {
        // Build a second tree and search.
        Timer::Start("tree_building");
        Log::Info << "Building query tree..." << endl;
        vector<size_t> oldFromNewQueries;
        RSType<tree::BallTree>::Tree queryTree(move(querySet),
            oldFromNewQueries, leafSize);
        Log::Info << "Tree built." << endl;
        Timer::Stop("tree_building");

        vector<vector<size_t>> neighborsOut;
        vector<vector<double>> distancesOut;
        ballTreeRS->Search(&queryTree, range, neighborsOut, distancesOut);

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
        // Search without building a second tree.
        ballTreeRS->Search(querySet, range, neighbors, distances);
      }
      break;

    case X_TREE:
      xTreeRS->Search(querySet, range, neighbors, distances);
      break;
  }
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

  switch (treeType)
  {
    case KD_TREE:
      kdTreeRS->Search(range, neighbors, distances);
      break;

    case COVER_TREE:
      coverTreeRS->Search(range, neighbors, distances);
      break;

    case R_TREE:
      rTreeRS->Search(range, neighbors, distances);
      break;

    case R_STAR_TREE:
      rStarTreeRS->Search(range, neighbors, distances);
      break;

    case BALL_TREE:
      ballTreeRS->Search(range, neighbors, distances);
      break;

    case X_TREE:
      xTreeRS->Search(range, neighbors, distances);
      break;
  }
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
    default:
      return "unknown tree";
  }
}

// Clean memory.
void RSModel::CleanMemory()
{
  if (kdTreeRS)
    delete kdTreeRS;
  if (coverTreeRS)
    delete coverTreeRS;
  if (rTreeRS)
    delete rTreeRS;
  if (rStarTreeRS)
    delete rStarTreeRS;
  if (ballTreeRS)
    delete ballTreeRS;
  if (xTreeRS)
    delete xTreeRS;

  kdTreeRS = NULL;
  coverTreeRS = NULL;
  rTreeRS = NULL;
  rStarTreeRS = NULL;
  ballTreeRS = NULL;
  xTreeRS = NULL;
}
