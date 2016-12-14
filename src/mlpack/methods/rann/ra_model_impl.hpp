/**
 * @file ra_model_impl.hpp
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

namespace mlpack {
namespace neighbor {

template<typename SortPolicy>
RAModel<SortPolicy>::RAModel(const TreeTypes treeType, const bool randomBasis) :
    treeType(treeType),
    leafSize(20),
    randomBasis(randomBasis),
    kdTreeRA(NULL),
    coverTreeRA(NULL),
    rTreeRA(NULL),
    rStarTreeRA(NULL),
    xTreeRA(NULL),
    hilbertRTreeRA(NULL),
    rPlusTreeRA(NULL),
    rPlusPlusTreeRA(NULL),
    ubTreeRA(NULL),
    octreeRA(NULL)
{
  // Nothing to do.
}

template<typename SortPolicy>
RAModel<SortPolicy>::~RAModel()
{
  delete kdTreeRA;
  delete coverTreeRA;
  delete rTreeRA;
  delete rStarTreeRA;
  delete xTreeRA;
  delete hilbertRTreeRA;
  delete rPlusTreeRA;
  delete rPlusPlusTreeRA;
  delete ubTreeRA;
  delete octreeRA;
}

template<typename SortPolicy>
template<typename Archive>
void RAModel<SortPolicy>::Serialize(Archive& ar,
                                    const unsigned int /* version */)
{
  ar & data::CreateNVP(treeType, "treeType");
  ar & data::CreateNVP(randomBasis, "randomBasis");
  ar & data::CreateNVP(q, "q");

  // This should never happen, but just in case, be clean with memory.
  if (Archive::is_loading::value)
  {
    delete kdTreeRA;
    delete coverTreeRA;
    delete rTreeRA;
    delete rStarTreeRA;
    delete xTreeRA;
    delete hilbertRTreeRA;
    delete rPlusTreeRA;
    delete rPlusPlusTreeRA;
    delete ubTreeRA;
    delete octreeRA;

    // Set all the pointers to NULL.
    kdTreeRA = NULL;
    coverTreeRA = NULL;
    rTreeRA = NULL;
    rStarTreeRA = NULL;
    xTreeRA = NULL;
    hilbertRTreeRA = NULL;
    rPlusPlusTreeRA = NULL;
    rPlusTreeRA = NULL;
    ubTreeRA = NULL;
  }

  // We only need to serialize one of the kRANN objects.
  switch (treeType)
  {
    case KD_TREE:
      ar & data::CreateNVP(kdTreeRA, "ra_model");
      break;
    case COVER_TREE:
      ar & data::CreateNVP(coverTreeRA, "ra_model");
      break;
    case R_TREE:
      ar & data::CreateNVP(rTreeRA, "ra_model");
      break;
    case R_STAR_TREE:
      ar & data::CreateNVP(rStarTreeRA, "ra_model");
      break;
    case X_TREE:
      ar & data::CreateNVP(xTreeRA, "ra_model");
      break;
    case HILBERT_R_TREE:
      ar & data::CreateNVP(hilbertRTreeRA, "ra_model");
      break;
    case R_PLUS_TREE:
      ar & data::CreateNVP(rPlusTreeRA, "ra_model");
      break;
    case R_PLUS_PLUS_TREE:
      ar & data::CreateNVP(rPlusPlusTreeRA, "ra_model");
      break;
    case UB_TREE:
      ar & data::CreateNVP(ubTreeRA, "ra_model");
      break;
    case OCTREE:
      ar & data::CreateNVP(octreeRA, "ra_model");
      break;
  }
}

template<typename SortPolicy>
const arma::mat& RAModel<SortPolicy>::Dataset() const
{
  if (kdTreeRA)
    return kdTreeRA->ReferenceSet();
  else if (coverTreeRA)
    return coverTreeRA->ReferenceSet();
  else if (rTreeRA)
    return rTreeRA->ReferenceSet();
  else if (rStarTreeRA)
    return rStarTreeRA->ReferenceSet();
  else if (xTreeRA)
    return xTreeRA->ReferenceSet();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->ReferenceSet();
  else if (rPlusTreeRA)
    return rPlusTreeRA->ReferenceSet();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->ReferenceSet();
  else if (ubTreeRA)
    return ubTreeRA->ReferenceSet();
  else if (octreeRA)
    return octreeRA->ReferenceSet();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::Naive() const
{
  if (kdTreeRA)
    return kdTreeRA->Naive();
  else if (coverTreeRA)
    return coverTreeRA->Naive();
  else if (rTreeRA)
    return rTreeRA->Naive();
  else if (rStarTreeRA)
    return rStarTreeRA->Naive();
  else if (xTreeRA)
    return xTreeRA->Naive();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->Naive();
  else if (rPlusTreeRA)
    return rPlusTreeRA->Naive();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->Naive();
  else if (ubTreeRA)
    return ubTreeRA->Naive();
  else if (octreeRA)
    return octreeRA->Naive();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::Naive()
{
  if (kdTreeRA)
    return kdTreeRA->Naive();
  else if (coverTreeRA)
    return coverTreeRA->Naive();
  else if (rTreeRA)
    return rTreeRA->Naive();
  else if (rStarTreeRA)
    return rStarTreeRA->Naive();
  else if (xTreeRA)
    return xTreeRA->Naive();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->Naive();
  else if (rPlusTreeRA)
    return rPlusTreeRA->Naive();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->Naive();
  else if (ubTreeRA)
    return ubTreeRA->Naive();
  else if (octreeRA)
    return octreeRA->Naive();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::SingleMode() const
{
  if (kdTreeRA)
    return kdTreeRA->SingleMode();
  else if (coverTreeRA)
    return coverTreeRA->SingleMode();
  else if (rTreeRA)
    return rTreeRA->SingleMode();
  else if (rStarTreeRA)
    return rStarTreeRA->SingleMode();
  else if (xTreeRA)
    return xTreeRA->SingleMode();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->SingleMode();
  else if (rPlusTreeRA)
    return rPlusTreeRA->SingleMode();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->SingleMode();
  else if (ubTreeRA)
    return ubTreeRA->SingleMode();
  else if (octreeRA)
    return octreeRA->SingleMode();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::SingleMode()
{
  if (kdTreeRA)
    return kdTreeRA->SingleMode();
  else if (coverTreeRA)
    return coverTreeRA->SingleMode();
  else if (rTreeRA)
    return rTreeRA->SingleMode();
  else if (rStarTreeRA)
    return rStarTreeRA->SingleMode();
  else if (xTreeRA)
    return xTreeRA->SingleMode();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->SingleMode();
  else if (rPlusTreeRA)
    return rPlusTreeRA->SingleMode();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->SingleMode();
  else if (ubTreeRA)
    return ubTreeRA->SingleMode();
  else if (octreeRA)
    return octreeRA->SingleMode();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
double RAModel<SortPolicy>::Tau() const
{
  if (kdTreeRA)
    return kdTreeRA->Tau();
  else if (coverTreeRA)
    return coverTreeRA->Tau();
  else if (rTreeRA)
    return rTreeRA->Tau();
  else if (rStarTreeRA)
    return rStarTreeRA->Tau();
  else if (xTreeRA)
    return xTreeRA->Tau();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->Tau();
  else if (rPlusTreeRA)
    return rPlusTreeRA->Tau();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->Tau();
  else if (ubTreeRA)
    return ubTreeRA->Tau();
  else if (octreeRA)
    return octreeRA->Tau();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
double& RAModel<SortPolicy>::Tau()
{
  if (kdTreeRA)
    return kdTreeRA->Tau();
  else if (coverTreeRA)
    return coverTreeRA->Tau();
  else if (rTreeRA)
    return rTreeRA->Tau();
  else if (rStarTreeRA)
    return rStarTreeRA->Tau();
  else if (xTreeRA)
    return xTreeRA->Tau();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->Tau();
  else if (rPlusTreeRA)
    return rPlusTreeRA->Tau();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->Tau();
  else if (ubTreeRA)
    return ubTreeRA->Tau();
  else if (octreeRA)
    return octreeRA->Tau();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
double RAModel<SortPolicy>::Alpha() const
{
  if (kdTreeRA)
    return kdTreeRA->Alpha();
  else if (coverTreeRA)
    return coverTreeRA->Alpha();
  else if (rTreeRA)
    return rTreeRA->Alpha();
  else if (rStarTreeRA)
    return rStarTreeRA->Alpha();
  else if (xTreeRA)
    return xTreeRA->Alpha();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->Alpha();
  else if (rPlusTreeRA)
    return rPlusTreeRA->Alpha();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->Alpha();
  else if (ubTreeRA)
    return ubTreeRA->Alpha();
  else if (octreeRA)
    return octreeRA->Alpha();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
double& RAModel<SortPolicy>::Alpha()
{
  if (kdTreeRA)
    return kdTreeRA->Alpha();
  else if (coverTreeRA)
    return coverTreeRA->Alpha();
  else if (rTreeRA)
    return rTreeRA->Alpha();
  else if (rStarTreeRA)
    return rStarTreeRA->Alpha();
  else if (xTreeRA)
    return xTreeRA->Alpha();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->Alpha();
  else if (rPlusTreeRA)
    return rPlusTreeRA->Alpha();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->Alpha();
  else if (ubTreeRA)
    return ubTreeRA->Alpha();
  else if (octreeRA)
    return octreeRA->Alpha();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::SampleAtLeaves() const
{
  if (kdTreeRA)
    return kdTreeRA->SampleAtLeaves();
  else if (coverTreeRA)
    return coverTreeRA->SampleAtLeaves();
  else if (rTreeRA)
    return rTreeRA->SampleAtLeaves();
  else if (rStarTreeRA)
    return rStarTreeRA->SampleAtLeaves();
  else if (xTreeRA)
    return xTreeRA->SampleAtLeaves();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->SampleAtLeaves();
  else if (rPlusTreeRA)
    return rPlusTreeRA->SampleAtLeaves();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->SampleAtLeaves();
  else if (ubTreeRA)
    return ubTreeRA->SampleAtLeaves();
  else if (octreeRA)
    return octreeRA->SampleAtLeaves();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::SampleAtLeaves()
{
  if (kdTreeRA)
    return kdTreeRA->SampleAtLeaves();
  else if (coverTreeRA)
    return coverTreeRA->SampleAtLeaves();
  else if (rTreeRA)
    return rTreeRA->SampleAtLeaves();
  else if (rStarTreeRA)
    return rStarTreeRA->SampleAtLeaves();
  else if (xTreeRA)
    return xTreeRA->SampleAtLeaves();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->SampleAtLeaves();
  else if (rPlusTreeRA)
    return rPlusTreeRA->SampleAtLeaves();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->SampleAtLeaves();
  else if (ubTreeRA)
    return ubTreeRA->SampleAtLeaves();
  else if (octreeRA)
    return octreeRA->SampleAtLeaves();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::FirstLeafExact() const
{
  if (kdTreeRA)
    return kdTreeRA->FirstLeafExact();
  else if (coverTreeRA)
    return coverTreeRA->FirstLeafExact();
  else if (rTreeRA)
    return rTreeRA->FirstLeafExact();
  else if (rStarTreeRA)
    return rStarTreeRA->FirstLeafExact();
  else if (xTreeRA)
    return xTreeRA->FirstLeafExact();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->FirstLeafExact();
  else if (rPlusTreeRA)
    return rPlusTreeRA->FirstLeafExact();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->FirstLeafExact();
  else if (ubTreeRA)
    return ubTreeRA->FirstLeafExact();
  else if (octreeRA)
    return octreeRA->FirstLeafExact();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::FirstLeafExact()
{
  if (kdTreeRA)
    return kdTreeRA->FirstLeafExact();
  else if (coverTreeRA)
    return coverTreeRA->FirstLeafExact();
  else if (rTreeRA)
    return rTreeRA->FirstLeafExact();
  else if (rStarTreeRA)
    return rStarTreeRA->FirstLeafExact();
  else if (xTreeRA)
    return xTreeRA->FirstLeafExact();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->FirstLeafExact();
  else if (rPlusTreeRA)
    return rPlusTreeRA->FirstLeafExact();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->FirstLeafExact();
  else if (ubTreeRA)
    return ubTreeRA->FirstLeafExact();
  else if (octreeRA)
    return octreeRA->FirstLeafExact();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
size_t RAModel<SortPolicy>::SingleSampleLimit() const
{
  if (kdTreeRA)
    return kdTreeRA->SingleSampleLimit();
  else if (coverTreeRA)
    return coverTreeRA->SingleSampleLimit();
  else if (rTreeRA)
    return rTreeRA->SingleSampleLimit();
  else if (rStarTreeRA)
    return rStarTreeRA->SingleSampleLimit();
  else if (xTreeRA)
    return xTreeRA->SingleSampleLimit();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->SingleSampleLimit();
  else if (rPlusTreeRA)
    return rPlusTreeRA->SingleSampleLimit();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->SingleSampleLimit();
  else if (ubTreeRA)
    return ubTreeRA->SingleSampleLimit();
  else if (octreeRA)
    return octreeRA->SingleSampleLimit();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
size_t& RAModel<SortPolicy>::SingleSampleLimit()
{
  if (kdTreeRA)
    return kdTreeRA->SingleSampleLimit();
  else if (coverTreeRA)
    return coverTreeRA->SingleSampleLimit();
  else if (rTreeRA)
    return rTreeRA->SingleSampleLimit();
  else if (rStarTreeRA)
    return rStarTreeRA->SingleSampleLimit();
  else if (xTreeRA)
    return xTreeRA->SingleSampleLimit();
  else if (hilbertRTreeRA)
    return hilbertRTreeRA->SingleSampleLimit();
  else if (rPlusTreeRA)
    return rPlusTreeRA->SingleSampleLimit();
  else if (rPlusPlusTreeRA)
    return rPlusPlusTreeRA->SingleSampleLimit();
  else if (ubTreeRA)
    return ubTreeRA->SingleSampleLimit();
  else if (octreeRA)
    return octreeRA->SingleSampleLimit();

  throw std::runtime_error("no rank-approximate nearest neighbor search model "
      "initialized");
}

template<typename SortPolicy>
size_t RAModel<SortPolicy>::LeafSize() const
{
  return leafSize;
}

template<typename SortPolicy>
size_t& RAModel<SortPolicy>::LeafSize()
{
  return leafSize;
}

template<typename SortPolicy>
typename RAModel<SortPolicy>::TreeTypes RAModel<SortPolicy>::TreeType() const
{
  return treeType;
}

template<typename SortPolicy>
typename RAModel<SortPolicy>::TreeTypes& RAModel<SortPolicy>::TreeType()
{
  return treeType;
}

template<typename SortPolicy>
bool RAModel<SortPolicy>::RandomBasis() const
{
  return randomBasis;
}

template<typename SortPolicy>
bool& RAModel<SortPolicy>::RandomBasis()
{
  return randomBasis;
}

template<typename SortPolicy>
void RAModel<SortPolicy>::BuildModel(arma::mat&& referenceSet,
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

  // Clean memory, if necessary.
  delete kdTreeRA;
  delete coverTreeRA;
  delete rTreeRA;
  delete rStarTreeRA;
  delete xTreeRA;
  delete hilbertRTreeRA;
  delete rPlusTreeRA;
  delete rPlusPlusTreeRA;
  delete ubTreeRA;
  delete octreeRA;

  this->leafSize = leafSize;

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
      // Build tree, if necessary.
      if (naive)
      {
        kdTreeRA = new RAType<tree::KDTree>(std::move(referenceSet), naive,
            singleMode);
      }
      else
      {
        std::vector<size_t> oldFromNewReferences;
        typename RAType<tree::KDTree>::Tree* kdTree =
            new typename RAType<tree::KDTree>::Tree(std::move(referenceSet),
            oldFromNewReferences, leafSize);
        kdTreeRA = new RAType<tree::KDTree>(kdTree, singleMode);

        // Give the model ownership of the tree.
        kdTreeRA->treeOwner = true;
        kdTreeRA->oldFromNewReferences = oldFromNewReferences;
      }
      break;
    case COVER_TREE:
      coverTreeRA = new RAType<tree::StandardCoverTree>(std::move(referenceSet),
          naive, singleMode);
      break;
    case R_TREE:
      rTreeRA = new RAType<tree::RTree>(std::move(referenceSet), naive,
          singleMode);
      break;
    case R_STAR_TREE:
      rStarTreeRA = new RAType<tree::RStarTree>(std::move(referenceSet), naive,
          singleMode);
      break;
    case X_TREE:
      xTreeRA = new RAType<tree::XTree>(std::move(referenceSet), naive,
          singleMode);
      break;
    case HILBERT_R_TREE:
      hilbertRTreeRA = new RAType<tree::HilbertRTree>(std::move(referenceSet),
          naive, singleMode);
      break;
    case R_PLUS_TREE:
      rPlusTreeRA = new RAType<tree::RPlusTree>(std::move(referenceSet),
          naive, singleMode);
      break;
    case R_PLUS_PLUS_TREE:
      rPlusPlusTreeRA = new RAType<tree::RPlusPlusTree>(std::move(referenceSet),
          naive, singleMode);
      break;
    case UB_TREE:
      ubTreeRA = new RAType<tree::UBTree>(std::move(referenceSet),
          naive, singleMode);
      break;
    case OCTREE:
      // Build tree, if necessary.
      if (naive)
      {
        octreeRA = new RAType<tree::Octree>(std::move(referenceSet), naive,
            singleMode);
      }
      else
      {
        std::vector<size_t> oldFromNewReferences;
        typename RAType<tree::Octree>::Tree* octree =
            new typename RAType<tree::Octree>::Tree(std::move(referenceSet),
            oldFromNewReferences, leafSize);
        octreeRA = new RAType<tree::Octree>(octree, singleMode);

        // Give the model ownership of the tree.
        octreeRA->treeOwner = true;
        octreeRA->oldFromNewReferences = oldFromNewReferences;
      }
      break;
  }

  if (!naive)
  {
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << std::endl;
  }
}

template<typename SortPolicy>
void RAModel<SortPolicy>::Search(arma::mat&& querySet,
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

  switch (treeType)
  {
    case KD_TREE:
      if (!kdTreeRA->Naive() && !kdTreeRA->SingleMode())
      {
        // Build a second tree and search.
        Timer::Start("tree_building");
        Log::Info << "Building query tree..." << std::endl;
        std::vector<size_t> oldFromNewQueries;
        typename RAType<tree::KDTree>::Tree queryTree(std::move(querySet),
            oldFromNewQueries, leafSize);
        Log::Info << "Tree built." << std::endl;
        Timer::Stop("tree_building");

        arma::Mat<size_t> neighborsOut;
        arma::mat distancesOut;
        kdTreeRA->Search(&queryTree, k, neighborsOut, distancesOut);

        // Unmap the query points.
        distances.set_size(distancesOut.n_rows, distancesOut.n_cols);
        neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
        for (size_t i = 0; i < neighborsOut.n_cols; ++i)
        {
          neighbors.col(oldFromNewQueries[i]) = neighborsOut.col(i);
          distances.col(oldFromNewQueries[i]) = distancesOut.col(i);
        }
      }
      else
      {
        // Search without building a second tree.
        kdTreeRA->Search(querySet, k, neighbors, distances);
      }
      break;
    case COVER_TREE:
      // No mapping necessary.
      coverTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case R_TREE:
      // No mapping necessary.
      rTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case R_STAR_TREE:
      // No mapping necessary.
      rStarTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case X_TREE:
      // No mapping necessary.
      xTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case HILBERT_R_TREE:
      // No mapping necessary.
      hilbertRTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case R_PLUS_TREE:
      // No mapping necessary.
      rPlusTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case R_PLUS_PLUS_TREE:
      // No mapping necessary.
      rPlusPlusTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case UB_TREE:
      // No mapping necessary.
      ubTreeRA->Search(querySet, k, neighbors, distances);
      break;
    case OCTREE:
      if (!octreeRA->Naive() && !octreeRA->SingleMode())
      {
        // Build a second tree and search.
        Timer::Start("tree_building");
        Log::Info << "Building query tree..." << std::endl;
        std::vector<size_t> oldFromNewQueries;
        typename RAType<tree::Octree>::Tree queryTree(std::move(querySet),
            oldFromNewQueries, leafSize);
        Log::Info << "Tree built." << std::endl;
        Timer::Stop("tree_building");

        arma::Mat<size_t> neighborsOut;
        arma::mat distancesOut;
        octreeRA->Search(&queryTree, k, neighborsOut, distancesOut);

        // Unmap the query points.
        distances.set_size(distancesOut.n_rows, distancesOut.n_cols);
        neighbors.set_size(neighborsOut.n_rows, neighborsOut.n_cols);
        for (size_t i = 0; i < neighborsOut.n_cols; ++i)
        {
          neighbors.col(oldFromNewQueries[i]) = neighborsOut.col(i);
          distances.col(oldFromNewQueries[i]) = distancesOut.col(i);
        }
      }
      else
      {
        // Search without building a second tree.
        octreeRA->Search(querySet, k, neighbors, distances);
      }
      break;
  }
}

template<typename SortPolicy>
void RAModel<SortPolicy>::Search(const size_t k,
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

  switch (treeType)
  {
    case KD_TREE:
      kdTreeRA->Search(k, neighbors, distances);
      break;
    case COVER_TREE:
      coverTreeRA->Search(k, neighbors, distances);
      break;
    case R_TREE:
      rTreeRA->Search(k, neighbors, distances);
      break;
    case R_STAR_TREE:
      rStarTreeRA->Search(k, neighbors, distances);
      break;
    case X_TREE:
      xTreeRA->Search(k, neighbors, distances);
      break;
    case HILBERT_R_TREE:
      hilbertRTreeRA->Search(k, neighbors, distances);
      break;
    case R_PLUS_TREE:
      rPlusTreeRA->Search(k, neighbors, distances);
      break;
    case R_PLUS_PLUS_TREE:
      rPlusPlusTreeRA->Search(k, neighbors, distances);
      break;
    case UB_TREE:
      ubTreeRA->Search(k, neighbors, distances);
      break;
    case OCTREE:
      octreeRA->Search(k, neighbors, distances);
      break;
  }
}

template<typename SortPolicy>
std::string RAModel<SortPolicy>::TreeName() const
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

#endif
