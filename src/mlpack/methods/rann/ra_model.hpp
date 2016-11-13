/**
 * @file ra_model.hpp
 * @author Ryan Curtin
 *
 * This is a model for rank-approximate nearest neighbor search.  It provides an
 * easy way to serialize a rank-approximate neighbor search model by abstracting
 * the types of trees and reflecting the RASearch API.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_MODEL_HPP
#define MLPACK_METHODS_RANN_RA_MODEL_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
#include <mlpack/core/tree/octree.hpp>

#include "ra_search.hpp"

namespace mlpack {
namespace neighbor {

/**
 * The RAModel class provides an abstraction for the RASearch class, abstracting
 * away the TreeType parameter and allowing it to be specified at runtime in
 * this class.  This class is written for the sake of the 'allkrann' program,
 * but is not necessarily restricted to that use.
 *
 * @param SortPolicy Sorting policy for neighbor searching (see RASearch).
 */
template<typename SortPolicy>
class RAModel
{
 public:
  /**
   * The list of tree types we can use with RASearch.  Does not include ball
   * trees; see #338.
   */
  enum TreeTypes
  {
    KD_TREE,
    COVER_TREE,
    R_TREE,
    R_STAR_TREE,
    X_TREE,
    HILBERT_R_TREE,
    R_PLUS_TREE,
    R_PLUS_PLUS_TREE,
    UB_TREE,
    OCTREE
  };

 private:
  //! The type of tree being used.
  TreeTypes treeType;
  //! The leaf size of the tree being used (useful only for the kd-tree).
  size_t leafSize;

  //! If true, randomly project into a new basis.
  bool randomBasis;
  //! The basis to project into.
  arma::mat q;

  //! Typedef the RASearch class we'll use.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using RAType = RASearch<SortPolicy,
                          metric::EuclideanDistance,
                          arma::mat,
                          TreeType>;

  //! Non-NULL if the kd-tree is used.
  RAType<tree::KDTree>* kdTreeRA;
  //! Non-NULL if the cover tree is used.
  RAType<tree::StandardCoverTree>* coverTreeRA;
  //! Non-NULL if the R tree is used.
  RAType<tree::RTree>* rTreeRA;
  //! Non-NULL if the R* tree is used.
  RAType<tree::RStarTree>* rStarTreeRA;
  //! Non-NULL if the X tree is used.
  RAType<tree::XTree>* xTreeRA;
  //! Non-NULL if the Hilbert R tree is used.
  RAType<tree::HilbertRTree>* hilbertRTreeRA;
  //! Non-NULL if the R+ tree is used.
  RAType<tree::RPlusTree>* rPlusTreeRA;
  //! Non-NULL if the R++ tree is used.
  RAType<tree::RPlusPlusTree>* rPlusPlusTreeRA;
  //! Non-NULL if the UB tree is used.
  RAType<tree::UBTree>* ubTreeRA;
  //! Non-NULL if the octree is used.
  RAType<tree::Octree>* octreeRA;

 public:
  /**
   * Initialize the RAModel with the given type and whether or not a random
   * basis should be used.
   */
  RAModel(TreeTypes treeType = TreeTypes::KD_TREE, bool randomBasis = false);

  //! Clean memory, if necessary.
  ~RAModel();

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Expose the dataset.
  const arma::mat& Dataset() const;

  //! Get whether or not single-tree search is being used.
  bool SingleMode() const;
  //! Modify whether or not single-tree search is being used.
  bool& SingleMode();

  //! Get whether or not naive search is being used.
  bool Naive() const;
  //! Modify whether or not naive search is being used.
  bool& Naive();

  //! Get the rank-approximation in percentile of the data.
  double Tau() const;
  //! Modify the rank-approximation in percentile of the data.
  double& Tau();

  //! Get the desired success probability.
  double Alpha() const;
  //! Modify the desired success probability.
  double& Alpha();

  //! Get whether or not sampling is done at the leaves.
  bool SampleAtLeaves() const;
  //! Modify whether or not sampling is done at the leaves.
  bool& SampleAtLeaves();

  //! Get whether or not we traverse to the first leaf without approximation.
  bool FirstLeafExact() const;
  //! Modify whether or not we traverse to the first leaf without approximation.
  bool& FirstLeafExact();

  //! Get the limit on the size of a node that can be approximated.
  size_t SingleSampleLimit() const;
  //! Modify the limit on the size of a node that can be approximation.
  size_t& SingleSampleLimit();

  //! Get the leaf size (only relevant when the kd-tree is used).
  size_t LeafSize() const;
  //! Modify the leaf size (only relevant when the kd-tree is used).
  size_t& LeafSize();

  //! Get the type of tree being used.
  TreeTypes TreeType() const;
  //! Modify the type of tree being used.
  TreeTypes& TreeType();

  //! Get whether or not a random basis is being used.
  bool RandomBasis() const;
  //! Modify whether or not a random basis is being used.  Be sure to rebuild
  //! the model using BuildModel().
  bool& RandomBasis();

  //! Build the reference tree.
  void BuildModel(arma::mat&& referenceSet,
                  const size_t leafSize,
                  const bool naive,
                  const bool singleMode);

  //! Perform rank-approximate neighbor search, taking ownership of the query
  //! set.
  void Search(arma::mat&& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * Perform rank-approximate neighbor search, using the reference set as the
   * query set.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Get the name of the tree type.
  std::string TreeName() const;
};

} // namespace neighbor
} // namespace mlpack

#include "ra_model_impl.hpp"

#endif
