/**
 * @file ns_model.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also reflects the NeighborSearch API and
 * automatically directs to the right tree type.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>

#include "neighbor_search.hpp"

namespace mlpack {
namespace neighbor {

template<typename SortPolicy>
struct NSModelName
{
  static const std::string Name() { return "neighbor_search_model"; }
};

template<>
struct NSModelName<NearestNeighborSort>
{
  static const std::string Name() { return "nearest_neighbor_search_model"; }
};

template<>
struct NSModelName<FurthestNeighborSort>
{
  static const std::string Name() { return "furthest_neighbor_search_model"; }
};

template<typename SortPolicy>
class NSModel
{
 public:
  enum TreeTypes
  {
    KD_TREE,
    COVER_TREE,
    R_TREE,
    R_STAR_TREE,
    X_TREE,
    BALL_TREE
  };

 private:
  int treeType;
  size_t leafSize;

  // For random projections.
  bool randomBasis;
  arma::mat q;

  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using NSType = NeighborSearch<SortPolicy,
                                metric::EuclideanDistance,
                                arma::mat,
                                TreeType,
                                TreeType<metric::EuclideanDistance,
                                    NeighborSearchStat<SortPolicy>,
                                    arma::mat>::template DualTreeTraverser>;

  // Only one of these pointers will be non-NULL.
  NSType<tree::KDTree>* kdTreeNS;
  NSType<tree::StandardCoverTree>* coverTreeNS;
  NSType<tree::RTree>* rTreeNS;
  NSType<tree::RStarTree>* rStarTreeNS;
  NSType<tree::XTree>* xTreeNS;
  NSType<tree::BallTree>* ballTreeNS;

 public:
  /**
   * Initialize the NSModel with the given type and whether or not a random
   * basis should be used.
   */
  NSModel(int treeType = TreeTypes::KD_TREE, bool randomBasis = false);

  //! Clean memory, if necessary.
  ~NSModel();

  //! Serialize the neighbor search model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Expose the dataset.
  const arma::mat& Dataset() const;

  //! Expose singleMode.
  bool SingleMode() const;
  bool& SingleMode();

  bool Naive() const;
  bool& Naive();

  size_t LeafSize() const { return leafSize; }
  size_t& LeafSize() { return leafSize; }

  int TreeType() const { return treeType; }
  int& TreeType() { return treeType; }

  bool RandomBasis() const { return randomBasis; }
  bool& RandomBasis() { return randomBasis; }

  //! Build the reference tree.
  void BuildModel(arma::mat&& referenceSet,
                  const size_t leafSize,
                  const bool naive,
                  const bool singleMode);

  //! Perform neighbor search.  The query set will be reordered.
  void Search(arma::mat&& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Perform neighbor search.
  void Search(const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  std::string TreeName() const;
};

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "ns_model_impl.hpp"

#endif
