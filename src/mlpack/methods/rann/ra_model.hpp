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
#include "ra_search.hpp"

namespace mlpack {

/**
 * RAWrapperBase is a base wrapper class for holding all RASearch types
 * supported by RAModel.  All RASearch type wrappers inherit from this class,
 * allowing a simple interface via inheritance for all the different types we
 * want to support.
 */
class RAWrapperBase
{
 public:
  //! Create the RAWrapperBase object.  The base class does not hold anything,
  //! so this constructor does nothing.
  RAWrapperBase() { }

  //! Create a new RAWrapperBase that is the same as this one.  This function
  //! will properly handle polymorphism.
  virtual RAWrapperBase* Clone() const = 0;

  //! Destruct the RAWrapperBase (nothing to do).
  virtual ~RAWrapperBase() { }

  //! Return a reference to the dataset.
  virtual const arma::mat& Dataset() const = 0;

  //! Get the single sample limit.
  virtual size_t SingleSampleLimit() const = 0;
  //! Modify the single sample limit.
  virtual size_t& SingleSampleLimit() = 0;

  //! Get whether to do exact search at the first leaf.
  virtual bool FirstLeafExact() const = 0;
  //! Modify whether to do exact search at the first leaf.
  virtual bool& FirstLeafExact() = 0;

  //! Get whether to do sampling at leaves.
  virtual bool SampleAtLeaves() const = 0;
  //! Modify whether to do sampling at leaves.
  virtual bool& SampleAtLeaves() = 0;

  //! Get the value of alpha.
  virtual double Alpha() const = 0;
  //! Modify the value of alpha.
  virtual double& Alpha() = 0;

  //! Get the value of tau.
  virtual double Tau() const = 0;
  //! Modify the value of tau.
  virtual double& Tau() = 0;

  //! Get whether single-tree search is being used.
  virtual bool SingleMode() const = 0;
  //! Modify whether single-tree search is being used.
  virtual bool& SingleMode() = 0;

  //! Get whether naive search is being used.
  virtual bool Naive() const = 0;
  //! Modify whether naive search is being used.
  virtual bool& Naive() = 0;

  //! Train the RASearch model with the given parameters.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize) = 0;

  //! Perform bichromatic rank-approximate nearest neighbor search (i.e. search
  //! with a separate query set).
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      const size_t leafSize) = 0;

  //! Perform monochromatic rank-approximate nearest neighbor search (i.e. a
  //! search with the reference set as the query set).
  virtual void Search(util::Timers& timers,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances) = 0;
};

/**
 * RAWrapper is a wrapper class for most RASearch types.
 */
template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
class RAWrapper : public RAWrapperBase
{
 public:
  //! Construct the RAWrapper object, initializing the internally-held RASearch
  //! object.
  RAWrapper(const bool singleMode, const bool naive) :
      ra(singleMode, naive)
  {
    // Nothing else to do.
  }

  //! Delete the RAWrapper object.
  virtual ~RAWrapper() { }

  //! Create a copy of this RAWrapper object.  This correctly handles
  //! polymorphism.
  virtual RAWrapper* Clone() const { return new RAWrapper(*this); }

  //! Get a reference to the reference set.
  const arma::mat& Dataset() const { return ra.ReferenceSet(); }

  //! Get the single sample limit.
  size_t SingleSampleLimit() const { return ra.SingleSampleLimit(); }
  //! Modify the single sample limit.
  size_t& SingleSampleLimit() { return ra.SingleSampleLimit(); }

  //! Get whether to do exact search at the first leaf.
  bool FirstLeafExact() const { return ra.FirstLeafExact(); }
  //! Modify whether to do exact search at the first leaf.
  bool& FirstLeafExact() { return ra.FirstLeafExact(); }

  //! Get whether to do sampling at leaves.
  bool SampleAtLeaves() const { return ra.SampleAtLeaves(); }
  //! Modify whether to do sampling at leaves.
  bool& SampleAtLeaves() { return ra.SampleAtLeaves(); }

  //! Get the value of alpha.
  double Alpha() const { return ra.Alpha(); }
  //! Modify the value of alpha.
  double& Alpha() { return ra.Alpha(); }

  //! Get the value of tau.
  double Tau() const { return ra.Tau(); }
  //! Modify the value of tau.
  double& Tau() { return ra.Tau(); }

  //! Get whether single-tree search is being used.
  bool SingleMode() const { return ra.SingleMode(); }
  //! Modify whether single-tree search is being used.
  bool& SingleMode() { return ra.SingleMode(); }

  //! Get whether naive search is being used.
  bool Naive() const { return ra.Naive(); }
  //! Modify whether naive search is being used.
  bool& Naive() { return ra.Naive(); }

  //! Train the model.  For RAWrapper, we ignore the leaf size.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t /* leafSize */);

  //! Perform bichromatic neighbor search (i.e. search with a separate query
  //! set).  For RAWrapper, we ignore the leaf size.
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      const size_t /* leafSize */);

  //! Perform monochromatic neighbor search (i.e. search where the reference set
  //! is used as the query set).
  virtual void Search(util::Timers& timers,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances);

  //! Serialize the RASearch model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(ra));
  }

 protected:
  using RAType = RASearch<NearestNeighborSort,
                          EuclideanDistance,
                          arma::mat,
                          TreeType>;

  //! The instantiated RASearch object that we are wrapping.
  RAType ra;
};

/**
 * LeafSizeRAWrapper wraps any RASearch type that needs to be able to take the
 * leaf size into account when building trees.  The implementations of Train()
 * and bichromatic Search() take this leaf size into account.
 */
template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
class LeafSizeRAWrapper : public RAWrapper<TreeType>
{
 public:
  //! Construct the LeafSizeRAWrapper by delegating to the RAWrapper
  //! constructor.
  LeafSizeRAWrapper(const bool singleMode, const bool naive) :
      RAWrapper<TreeType>(singleMode, naive)
  {
    // Nothing else to do.
  }

  //! Delete the LeafSizeRAWrapper.
  virtual ~LeafSizeRAWrapper() { }

  //! Return a copy of the LeafSizeRAWrapper.
  virtual LeafSizeRAWrapper* Clone() const
  {
    return new LeafSizeRAWrapper(*this);
  }

  //! Train a model with the given parameters.  This overload uses leafSize.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize);

  //! Perform bichromatic search (e.g. search with a separate query set).  This
  //! overload takes the leaf size into account to build the query tree.
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      const size_t leafSize);

  //! Serialize the RASearch model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(ra));
  }

 protected:
  using RAWrapper<TreeType>::ra;
};

/**
 * The RAModel class provides an abstraction for the RASearch class, abstracting
 * away the TreeType parameter and allowing it to be specified at runtime in
 * this class.  This class is written for the sake of the 'allkrann' program,
 * but is not necessarily restricted to that use.
 */
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
  RAWrapperBase* raSearch;

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
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Expose the dataset.
  const arma::mat& Dataset() const { return raSearch->Dataset(); }

  //! Get whether or not single-tree search is being used.
  bool SingleMode() const { return raSearch->SingleMode(); }
  //! Modify whether or not single-tree search is being used.
  bool& SingleMode() { return raSearch->SingleMode(); }

  //! Get whether or not naive search is being used.
  bool Naive() const { return raSearch->Naive(); }
  //! Modify whether or not naive search is being used.
  bool& Naive() { return raSearch->Naive(); }

  //! Get the rank-approximation in percentile of the data.
  double Tau() const { return raSearch->Tau(); }
  //! Modify the rank-approximation in percentile of the data.
  double& Tau() { return raSearch->Tau(); }

  //! Get the desired success probability.
  double Alpha() const { return raSearch->Alpha(); }
  //! Modify the desired success probability.
  double& Alpha() { return raSearch->Alpha(); }

  //! Get whether or not sampling is done at the leaves.
  bool SampleAtLeaves() const { return raSearch->SampleAtLeaves(); }
  //! Modify whether or not sampling is done at the leaves.
  bool& SampleAtLeaves() { return raSearch->SampleAtLeaves(); }

  //! Get whether or not we traverse to the first leaf without approximation.
  bool FirstLeafExact() const { return raSearch->FirstLeafExact(); }
  //! Modify whether or not we traverse to the first leaf without approximation.
  bool& FirstLeafExact() { return raSearch->FirstLeafExact(); }

  //! Get the limit on the size of a node that can be approximated.
  size_t SingleSampleLimit() const { return raSearch->SingleSampleLimit(); }
  //! Modify the limit on the size of a node that can be approximation.
  size_t& SingleSampleLimit() { return raSearch->SingleSampleLimit(); }

  //! Get the leaf size (only relevant when the kd-tree is used).
  size_t LeafSize() const { return leafSize; }
  //! Modify the leaf size (only relevant when the kd-tree is used).
  size_t& LeafSize() { return leafSize; }

  //! Get the type of tree being used.
  TreeTypes TreeType() const { return treeType; }
  //! Modify the type of tree being used.
  TreeTypes& TreeType() { return treeType; }

  //! Get whether or not a random basis is being used.
  bool RandomBasis() const { return randomBasis; }
  //! Modify whether or not a random basis is being used.  Be sure to rebuild
  //! the model using BuildModel().
  bool& RandomBasis() { return randomBasis; }

  //! Initialize the model's memory.
  void InitializeModel(const bool naive, const bool singleMode);

  //! Build the reference tree.
  void BuildModel(util::Timers& timers,
                  arma::mat&& referenceSet,
                  const size_t leafSize,
                  const bool naive,
                  const bool singleMode);

  //! Perform rank-approximate neighbor search, taking ownership of the query
  //! set.
  void Search(util::Timers& timers,
              arma::mat&& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * Perform rank-approximate neighbor search, using the reference set as the
   * query set.
   */
  void Search(util::Timers& timers,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Get the name of the tree type.
  std::string TreeName() const;
};

} // namespace mlpack

#include "ra_model_impl.hpp"

#endif
