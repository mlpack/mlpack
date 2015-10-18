/**
 * @file ns_model_impl.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also reflects the NeighborSearch API and
 * automatically directs to the right tree type.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_IMPL_HPP

// In case it hasn't been included yet.
#include "ns_model.hpp"

namespace mlpack {
namespace neighbor {

/**
 * Initialize the NSModel with the given type and whether or not a random
 * basis should be used.
 */
template<typename SortPolicy>
NSModel<SortPolicy>::NSModel(int treeType, bool randomBasis) :
    treeType(treeType),
    randomBasis(randomBasis),
    kdTreeNS(NULL),
    coverTreeNS(NULL),
    rTreeNS(NULL),
    rStarTreeNS(NULL),
    kdTree(NULL)
{
  // Nothing to do.
}

//! Clean memory, if necessary.
template<typename SortPolicy>
NSModel<SortPolicy>::~NSModel()
{
  if (kdTree)
    delete kdTree;

  if (kdTreeNS)
    delete kdTreeNS;
  if (coverTreeNS)
    delete coverTreeNS;
  if (rTreeNS)
    delete rTreeNS;
  if (rStarTreeNS)
    delete rStarTreeNS;
}

//! Serialize the kNN model.
template<typename SortPolicy>
  template<typename Archive>
void NSModel<SortPolicy>::Serialize(Archive& ar, 
                                    const unsigned int /* version */)
{
  ar & data::CreateNVP(treeType, "treeType");
  ar & data::CreateNVP(randomBasis, "randomBasis");
  ar & data::CreateNVP(q, "q");
  ar & data::CreateNVP(oldFromNewReferences, "oldFromNewReferences");

  // This should never happen, but just in case, be clean with memory.
  if (Archive::is_loading::value)
  {
    if (kdTree)
      delete kdTree;

    if (kdTreeNS)
      delete kdTreeNS;
    if (coverTreeNS)
      delete coverTreeNS;
    if (rTreeNS)
      delete rTreeNS;
    if (rStarTreeNS)
      delete rStarTreeNS;

    // Set all the pointers to NULL.
    kdTree = NULL;

    kdTreeNS = NULL;
    coverTreeNS = NULL;
    rTreeNS = NULL;
    rStarTreeNS = NULL;
  }

  // We'll only need to serialize one of the kNN objects, based on the type.
  const std::string& name = NSModelName<SortPolicy>::value;
  switch (treeType)
  {
    case KD_TREE:
      ar & data::CreateNVP(kdTreeNS, name);
      break;
    case COVER_TREE:
      ar & data::CreateNVP(coverTreeNS, name);
      break;
    case R_TREE:
      ar & data::CreateNVP(rTreeNS, name);
      break;
    case R_STAR_TREE:
      ar & data::CreateNVP(rStarTreeNS, name);
      break;
  }
}

//! Expose singleMode.
template<typename SortPolicy>
bool NSModel<SortPolicy>::SingleMode() const
{
  if (kdTreeNS)
    return kdTreeNS->SingleMode();
  else if (coverTreeNS)
    return coverTreeNS->SingleMode();
  else if (rTreeNS)
    return rTreeNS->SingleMode();
  else if (rStarTreeNS)
    return rStarTreeNS->SingleMode();

  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
bool& NSModel<SortPolicy>::SingleMode()
{
  if (kdTreeNS)
    return kdTreeNS->SingleMode();
  else if (coverTreeNS)
    return coverTreeNS->SingleMode();
  else if (rTreeNS)
    return rTreeNS->SingleMode();
  else if (rStarTreeNS)
    return rStarTreeNS->SingleMode();

  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
bool NSModel<SortPolicy>::Naive() const
{
  if (kdTreeNS)
    return kdTreeNS->Naive();
  else if (coverTreeNS)
    return coverTreeNS->Naive();
  else if (rTreeNS)
    return rTreeNS->Naive();
  else if (rStarTreeNS)
    return rStarTreeNS->Naive();

  throw std::runtime_error("no neighbor search model initialized");
}

template<typename SortPolicy>
bool& NSModel<SortPolicy>::Naive()
{
  if (kdTreeNS)
    return kdTreeNS->Naive();
  else if (coverTreeNS)
    return coverTreeNS->Naive();
  else if (rTreeNS)
    return rTreeNS->Naive();
  else if (rStarTreeNS)
    return rStarTreeNS->Naive();

  throw std::runtime_error("no neighbor search model initialized");
}

//! Build the reference tree.
template<typename SortPolicy>
void NSModel<SortPolicy>::BuildModel(arma::mat& referenceSet,
                                     const size_t leafSize,
                                     const bool naive,
                                     const bool singleMode)
{
  // Initialize random basis if necessary.
  if (randomBasis)
  {
    Log::Info << "Creating random basis..." << std::endl;
    while (true)
    {
      // [Q, R] = qr(randn(d, d));
      // Q = Q * diag(sign(diag(R)));
      arma::mat r;
      if (arma::qr(q, r, arma::randn<arma::mat>(referenceSet.n_rows,
              referenceSet.n_rows)))
      {
        arma::vec rDiag(r.n_rows);
        for (size_t i = 0; i < rDiag.n_elem; ++i)
        {
          if (r(i, i) < 0)
            rDiag(i) = -1;
          else if (r(i, i) > 0)
            rDiag(i) = 1;
          else
            rDiag(i) = 0;
        }

        q *= arma::diagmat(rDiag);

        // Check if the determinant is positive.
        if (arma::det(q) >= 0)
          break;
      }
    }
  }

  // Clean memory, if necessary.
  if (kdTree)
    delete kdTree;

  if (kdTreeNS)
    delete kdTreeNS;
  if (coverTreeNS)
    delete coverTreeNS;
  if (rTreeNS)
    delete rTreeNS;
  if (rStarTreeNS)
    delete rStarTreeNS;

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
      // If necessary, build the kd-tree.
      if (naive)
      {
        kdTreeNS = new NSType<tree::KDTree>(referenceSet, naive, singleMode);
      }
      else
      {
        kdTree = new typename NSType<tree::KDTree>::Tree(referenceSet,
            oldFromNewReferences, leafSize);
        kdTreeNS = new NSType<tree::KDTree>(kdTree, singleMode);
      }

      break;
    case COVER_TREE:
      // If necessary, build the cover tree.
      coverTreeNS = new NSType<tree::StandardCoverTree>(referenceSet,
          singleMode);
      break;
    case R_TREE:
      // If necessary, build the R tree.
      rTreeNS = new NSType<tree::RTree>(referenceSet, singleMode);
      break;
    case R_STAR_TREE:
      // If necessary, build the R* tree.
      rStarTreeNS = new NSType<tree::RStarTree>(referenceSet, singleMode);
      break;
  }

  if (!naive)
  {
    Timer::Stop("tree_building");
    Log::Info << "Tree built." << std::endl;
  }
}

//! Perform neighbor search.  The query set will be reordered.
template<typename SortPolicy>
void NSModel<SortPolicy>::Search(arma::mat& querySet,
                                 const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  // We may need to map the query set randomly.
  if (randomBasis)
    querySet = q * querySet;

  Log::Info << "Searching for " << k << " nearest neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree search..." << std::endl;
  else if (!Naive())
    Log::Info << "single-tree search..." << std::endl;
  else
    Log::Info << "brute-force (naive) search..." << std::endl;

  switch (treeType)
  {
    case KD_TREE:
      if (!kdTreeNS->Naive() && !kdTreeNS->SingleMode())
      {
        // Build a second tree and search.
        Timer::Start("tree_building");
        Log::Info << "Building query tree..." << std::endl;
        std::vector<size_t> oldFromNewQueries;
        typename NSType<tree::KDTree>::Tree queryTree(querySet,
            oldFromNewQueries, leafSize);
        Log::Info << "Tree built." << std::endl;
        Timer::Stop("tree_building");

        arma::Mat<size_t> neighborsOut;
        arma::mat distancesOut;
        kdTreeNS->Search(&queryTree, k, neighborsOut, distancesOut);

        // Unmap the results.
        Unmap(neighborsOut, distancesOut, oldFromNewReferences,
            oldFromNewQueries, neighbors, distances);
      }
      else if (kdTreeNS->SingleMode() && !kdTreeNS->Naive())
      {
        // Search without building a second tree.
        arma::Mat<size_t> neighborsOut;
        arma::mat distancesOut;
        kdTreeNS->Search(querySet, k, neighborsOut, distancesOut);

        Unmap(neighborsOut, distancesOut, oldFromNewReferences, neighbors,
            distances);
      }
      else
      {
        // Naive mode search.  No unmapping will be necessary... unless a tree
        // has been built.
        if (oldFromNewReferences.size() == 0)
        {
          kdTreeNS->Search(querySet, k, neighbors, distances);
        }
        else
        {
          arma::Mat<size_t> neighborsOut;
          arma::mat distancesOut;
          kdTreeNS->Search(querySet, k, neighborsOut, distancesOut);

          Unmap(neighborsOut, distancesOut, oldFromNewReferences, neighbors,
              distances);
        }
      }
      break;
    case COVER_TREE:
      // No mapping necessary.
      coverTreeNS->Search(querySet, k, neighbors, distances);
      break;
    case R_TREE:
      // No mapping necessary.
      rTreeNS->Search(querySet, k, neighbors, distances);
      break;
    case R_STAR_TREE:
      // No mapping necessary.
      rStarTreeNS->Search(querySet, k, neighbors, distances);
      break;
  }
}

//! Perform neighbor search.
template<typename SortPolicy>
void NSModel<SortPolicy>::Search(const size_t k,
                                 arma::Mat<size_t>& neighbors,
                                 arma::mat& distances)
{
  Log::Info << "Searching for " << k << " nearest neighbors with ";
  if (!Naive() && !SingleMode())
    Log::Info << "dual-tree search..." << std::endl;
  else if (!Naive())
    Log::Info << "single-tree search..." << std::endl;
  else
    Log::Info << "brute-force (naive) search..." << std::endl;

  switch (treeType)
  {
    case KD_TREE:
      // If in dual-tree or single-tree mode, we'll have to do unmapping.  We
      // also must do unmapping in naive mode, if a tree has been built on the
      // data.
      if (oldFromNewReferences.size() > 0) // Mapping has occured.
      {
        arma::Mat<size_t> neighborsOut;
        arma::mat distancesOut;
        kdTreeNS->Search(k, neighborsOut, distancesOut);
        Unmap(neighborsOut, distancesOut, oldFromNewReferences,
            oldFromNewReferences, neighbors, distances);
      }
      else
      {
        kdTreeNS->Search(k, neighbors, distances);
      }
      break;
    case COVER_TREE:
      coverTreeNS->Search(k, neighbors, distances);
      break;
    case R_TREE:
      rTreeNS->Search(k, neighbors, distances);
      break;
    case R_STAR_TREE:
      rStarTreeNS->Search(k, neighbors, distances);
      break;
  }
}

} // namespace neighbor
} // namespace mlpack

#endif
