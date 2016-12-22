/**
 * @file rs_model.hpp
 * @author Ryan Curtin
 *
 * This is a model for range search.  It is useful in that it provides an easy
 * way to serialize a model, abstracts away the different types of trees, and
 * also reflects the RangeSearch API and automatically directs to the right
 * tree types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
#include <mlpack/core/tree/octree.hpp>

#include "range_search.hpp"

namespace mlpack {
namespace range {

class RSModel
{
 public:
  enum TreeTypes
  {
    KD_TREE,
    COVER_TREE,
    R_TREE,
    R_STAR_TREE,
    BALL_TREE,
    X_TREE,
    HILBERT_R_TREE,
    R_PLUS_TREE,
    R_PLUS_PLUS_TREE,
    VP_TREE,
    RP_TREE,
    MAX_RP_TREE,
    UB_TREE,
    OCTREE
  };

 private:
  TreeTypes treeType;
  size_t leafSize;

  //! If true, we randomly project the data into a new basis before search.
  bool randomBasis;
  //! Random projection matrix.
  arma::mat q;

  //! The mostly-specified type of the range search model.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using RSType = RangeSearch<metric::EuclideanDistance, arma::mat, TreeType>;

  // Only one of these pointers will be non-NULL.
  //! kd-tree based range search object (NULL if not in use).
  RSType<tree::KDTree>* kdTreeRS;
  //! Cover tree based range search object (NULL if not in use).
  RSType<tree::StandardCoverTree>* coverTreeRS;
  //! R tree based range search object (NULL if not in use).
  RSType<tree::RTree>* rTreeRS;
  //! R* tree based range search object (NULL if not in use).
  RSType<tree::RStarTree>* rStarTreeRS;
  //! Ball tree based range search object (NULL if not in use).
  RSType<tree::BallTree>* ballTreeRS;
  //! X tree based range search object (NULL if not in use).
  RSType<tree::XTree>* xTreeRS;
  //! Hilbert R tree based range search object (NULL if not in use).
  RSType<tree::HilbertRTree>* hilbertRTreeRS;
  //! R+ tree based range search object (NULL if not in use).
  RSType<tree::RPlusTree>* rPlusTreeRS;
  //! R++ tree based range search object (NULL if not in use).
  RSType<tree::RPlusPlusTree>* rPlusPlusTreeRS;
  //! VP tree based range search object (NULL if not in use).
  RSType<tree::VPTree>* vpTreeRS;
  //! Random projection tree (mean) based range search object
  //! (NULL if not in use).
  RSType<tree::RPTree>* rpTreeRS;
  //! Random projection tree (max) based range search object
  //! (NULL if not in use).
  RSType<tree::MaxRPTree>* maxRPTreeRS;
  //! Universal B tree based range search object
  //! (NULL if not in use).
  RSType<tree::UBTree>* ubTreeRS;
  //! Octree-based range search object (NULL if not in use).
  RSType<tree::Octree>* octreeRS;

 public:
  /**
   * Initialize the RSModel with the given type and whether or not a random
   * basis should be used.
   *
   * @param treeType Type of tree to use.
   * @param randomBasis Whether or not to use a random basis.
   */
  RSModel(const TreeTypes treeType = TreeTypes::KD_TREE,
          const bool randomBasis = false);

  /**
   * Clean memory, if necessary.
   */
  ~RSModel();

  //! Serialize the range search model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

  //! Expose the dataset.
  const arma::mat& Dataset() const;

  //! Get whether the model is in single-tree search mode.
  bool SingleMode() const;
  //! Modify whether the model is in single-tree search mode.
  bool& SingleMode();

  //! Get whether the model is in naive search mode.
  bool Naive() const;
  //! Modify whether the model is in naive search mode.
  bool& Naive();

  //! Get the leaf size (applicable to everything but the cover tree).
  size_t LeafSize() const { return leafSize; }
  //! Modify the leaf size (applicable to everything but the cover tree).
  size_t& LeafSize() { return leafSize; }

  //! Get the type of tree.
  TreeTypes TreeType() const { return treeType; }
  //! Modify the type of tree (don't do this after the model has been built).
  TreeTypes& TreeType() { return treeType; }

  //! Get whether a random basis is used.
  bool RandomBasis() const { return randomBasis; }
  //! Modify whether a random basis is used (don't do this after the model has
  //! been built).
  bool& RandomBasis() { return randomBasis; }

  /**
   * Build the reference tree on the given dataset with the given parameters.
   * This takes possession of the reference set to avoid a copy.
   *
   * @param referenceSet Set of reference points.
   * @param leafSize Leaf size of tree (ignored for the cover tree).
   * @param naive Whether naive search should be used.
   * @param singleMode Whether single-tree search should be used.
   */
  void BuildModel(arma::mat&& referenceSet,
                  const size_t leafSize,
                  const bool naive,
                  const bool singleMode);

  /**
   * Perform range search.  This takes possession of the query set, so the query
   * set will not be usable after the search.  For more information on the
   * output format, see RangeSearch<>::Search().
   *
   * @param querySet Set of query points.
   * @param range Range to search for.
   * @param neighbors Output: neighbors falling within the desired range.
   * @param distances Output: distances of neighbors.
   */
  void Search(arma::mat&& querySet,
              const math::Range& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<double>>& distances);

  /**
   * Perform monochromatic range search, with the reference set as the query
   * set.  For more information on the output format, see
   * RangeSearch<>::Search().
   *
   * @param range Range to search for.
   * @param neighbors Output: neighbors falling within the desired range.
   * @param distances Output: distances of neighbors.
   */
  void Search(const math::Range& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<double>>& distances);

 private:
  /**
   * Return a string representing the name of the tree.  This is used for
   * logging output.
   */
  std::string TreeName() const;

  /**
   * Clean up memory.
   */
  void CleanMemory();
};

} // namespace range
} // namespace mlpack

// Include implementation (of Serialize() and inline functions).
#include "rs_model_impl.hpp"

#endif
