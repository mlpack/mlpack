/**
 * @file methods/rann/ra_model.hpp
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
#include <boost/variant.hpp>
#include "ra_search.hpp"

namespace mlpack {
namespace neighbor {

/**
 * Alias template for RASearch
 */
template<typename SortPolicy,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
using RAType = RASearch<SortPolicy,
                        metric::EuclideanDistance,
                        arma::mat,
                        TreeType>;

/**
 * MonoSearchVisitor executes a monochromatic neighbor search on the given
 * RAType. We don't make any difference for different instantiation of RAType.
 */
class MonoSearchVisitor : public boost::static_visitor<void>
{
 private:
  //! Number of neighbors to search for.
  const size_t k;
  //! Result matrix for neighbors.
  arma::Mat<size_t>& neighbors;
  //! Result matrix for distances.
  arma::mat& distances;

 public:
  //! Perform monochromatic nearest neighbor search.
  template<typename RAType>
  void operator()(RAType* ra) const;

  //! Construct the MonoSearchVisitor object with the given parameters.
  MonoSearchVisitor(const size_t k,
                    arma::Mat<size_t>& neighbors,
                    arma::mat& distances) :
      k(k),
      neighbors(neighbors),
      distances(distances)
  {};
};

/**
 * BiSearchVisitor executes a bichromatic neighbor search on the given RAType.
 * We use template specialization to differentiate those tree types types that
 * accept leafSize as a parameter. In these cases, before doing neighbor search
 * a query tree with proper leafSize is built from the querySet.
 */
template<typename SortPolicy>
class BiSearchVisitor : public boost::static_visitor<void>
{
 private:
  //! The query set for the bichromatic search.
  const arma::mat& querySet;
  //! The number of neighbors to search for.
  const size_t k;
  //! The results matrix for neighbors.
  arma::Mat<size_t>& neighbors;
  //! The result matrix for distances.
  arma::mat& distances;
  //! The number of points in a leaf (for BinarySpaceTrees).
  const size_t leafSize;

  //! Bichromatic neighbor search on the given RAType considering leafSize.
  template<typename RAType>
  void SearchLeaf(RAType* ra) const;

 public:
  //! Alias template necessary for visual c++ compiler.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using RATypeT = RAType<SortPolicy, TreeType>;

  //! Default Bichromatic neighbor search on the given RAType instance.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(RATypeT<TreeType>* ra) const;

  //! Bichromatic search on the given RAType specialized for KDTrees.
  void operator()(RATypeT<tree::KDTree>* ra) const;

  //! Bichromatic search on the given RAType specialized for octrees.
  void operator()(RATypeT<tree::Octree>* ra) const;

  //! Construct the BiSearchVisitor.
  BiSearchVisitor(const arma::mat& querySet,
                  const size_t k,
                  arma::Mat<size_t>& neighbors,
                  arma::mat& distances,
                  const size_t leafSize);
};

/**
 * TrainVisitor sets the reference set to a new reference set on the given
 * RAType. We use template specialization to differentiate those trees that 
 * accept leafSize as a parameter. In these cases, a reference tree with proper
 * leafSize is built from the referenceSet.
 */
template<typename SortPolicy>
class TrainVisitor : public boost::static_visitor<void>
{
 private:
  //! The reference set to use for training.
  arma::mat&& referenceSet;
  //! The leaf size, used only by BinarySpaceTree.
  size_t leafSize;

  //! Train on the given RAType considering the leafSize.
  template<typename RAType>
  void TrainLeaf(RAType* ra) const;

 public:
  //! Alias template necessary for visual c++ compiler.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using RATypeT = RAType<SortPolicy, TreeType>;

  //! Default Train on the given RAType instance.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(RATypeT<TreeType>* ra) const;

  //! Train on the given RAType specialized for KDTrees.
  void operator()(RATypeT<tree::KDTree>* ra) const;

  //! Train on the given RAType specialized for Octrees.
  void operator()(RATypeT<tree::Octree>* ra) const;

  //! Construct the TrainVisitor object with the given reference set, leafSize
  //! for BinarySpaceTrees.
  TrainVisitor(arma::mat&& referenceSet,
               const size_t leafSize);
};

/**
 * Exposes the SingleSampleLimit() method of the given RAType.
 */
class SingleSampleLimitVisitor : public boost::static_visitor<size_t&>
{
 public:
  template<typename RAType>
  size_t& operator()(RAType* ra) const;
};

/**
 * Exposes the FirstLeafExact() method of the given RAType.
 */
class FirstLeafExactVisitor : public boost::static_visitor<bool&>
{
 public:
  template<typename RAType>
  bool& operator()(RAType* ra) const;
};

/**
 * Exposes the SampleAtLeaves() method of the given RAType.
 */
class SampleAtLeavesVisitor : public boost::static_visitor<bool&>
{
 public:
  //! Return SampleAtLeaves (whether or not sampling is done at leaves).
  template<typename RAType>
  bool& operator()(RAType *) const;
};

/**
 * Exposes the Alpha() method of the given RAType.
 */
class AlphaVisitor : public boost::static_visitor<double&>
{
 public:
  //! Return Alpha parameter.
  template<typename RAType>
  double& operator()(RAType* ra) const;
};

/**
 * Exposes the Tau() method of the given RAType.
 */
class TauVisitor : public boost::static_visitor<double&>
{
 public:
  //! Get a reference to the Tau parameter.
  template<typename RAType>
  double& operator()(RAType* ra) const;
};

/**
 * Exposes the SingleMode() method of the given RAType.
 */
class SingleModeVisitor : public boost::static_visitor<bool&>
{
 public:
  //! Get a reference to the SingleMode parameter of the given RASearch object.
  template<typename RAType>
  bool& operator()(RAType* ra) const;
};

/**
 * Exposes the referenceSet of the given RAType.
 */
class ReferenceSetVisitor : public boost::static_visitor<const arma::mat&>
{
 public:
  //! Return the reference set.
  template<typename RAType>
  const arma::mat& operator()(RAType* ra) const;
};

/**
 * DeleteVisitor deletes the give RAType Instance.
 */
class DeleteVisitor : public boost::static_visitor<void>
{
 public:
  //! Delete the RAType Object.
  template<typename RAType> void operator()(RAType* ra) const;
};

/**
 * NaiveVisitor exposes the Naive() method of the given RAType.
 */
class NaiveVisitor : public boost::static_visitor<bool&>
{
 public:
  /**
   * Get a reference to the naive parameter of the given RASearch object.
   */
  template<typename RAType>
  bool& operator()(RAType* ra) const;
};

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

  //! The rank-approximate model.
  boost::variant<RAType<SortPolicy, tree::KDTree>*,
                 RAType<SortPolicy, tree::StandardCoverTree>*,
                 RAType<SortPolicy, tree::RTree>*,
                 RAType<SortPolicy, tree::RStarTree>*,
                 RAType<SortPolicy, tree::XTree>*,
                 RAType<SortPolicy, tree::HilbertRTree>*,
                 RAType<SortPolicy, tree::RPlusTree>*,
                 RAType<SortPolicy, tree::RPlusPlusTree>*,
                 RAType<SortPolicy, tree::UBTree>*,
                 RAType<SortPolicy, tree::Octree>*> raSearch;

 public:
  /**
   * Initialize the RAModel with the given type and whether or not a random
   * basis should be used.
   */
  RAModel(TreeTypes treeType = TreeTypes::KD_TREE, bool randomBasis = false);

  /**
   * Copy the given RAModel.
   *
   * @param other RAModel to copy.
   */
  RAModel(const RAModel& other);

  /**
   * Take ownership of the given RAModel.
   *
   * @param other RAModel to take ownership of.
   */
  RAModel(RAModel&& other);

  /**
   * Copy the given RAModel.
   *
   * @param other RAModel to copy.
   */
  RAModel& operator=(const RAModel& other);

  /**
   * Take ownership of the given RAModel.
   *
   * @param other RAModel to take ownership of.
   */
  RAModel& operator=(RAModel&& other);

  //! Clean memory, if necessary.
  ~RAModel();

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

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
