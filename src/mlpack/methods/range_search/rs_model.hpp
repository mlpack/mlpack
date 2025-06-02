/**
 * @file methods/range_search/rs_model.hpp
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

/**
 * RSWrapperBase is a base wrapper class for holding all RangeSearch types
 * supported by RSModel.  All RangeSearch type wrappers inherit from this class,
 * allowing a simple interface via inheritance for all the different types we
 * want to support.
 */
class RSWrapperBase
{
 public:
  //! Create the RSWrapperBase object.  The base class does not hold anything,
  //! so this constructor does nothing.
  RSWrapperBase() { }

  //! Create a new RSWrapperBase that is the same as this one.  This function
  //! will properly handle polymorphism.
  virtual RSWrapperBase* Clone() const = 0;

  //! Destruct the RSWrapperBase (nothing to do).
  virtual ~RSWrapperBase() { }

  //! Get the dataset.
  virtual const arma::mat& Dataset() const = 0;

  //! Get whether single-tree search is being used.
  virtual bool SingleMode() const = 0;
  //! Modify whether single-tree search is being used.
  virtual bool& SingleMode() = 0;

  //! Get whether naive search is being used.
  virtual bool Naive() const = 0;
  //! Modify whether naive search is being used.
  virtual bool& Naive() = 0;

  //! Train the model (build the reference tree if needed).
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize) = 0;

  //! Perform bichromatic range search (i.e. a search with a separate query
  //! set).
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const Range& range,
                      std::vector<std::vector<size_t>>& neighbors,
                      std::vector<std::vector<double>>& distances,
                      const size_t leafSize) = 0;

  //! Perform monochromatic range search (i.e. a search with the reference set
  //! as the query set).
  virtual void Search(util::Timers& timers,
                      const Range& range,
                      std::vector<std::vector<size_t>>& neighbors,
                      std::vector<std::vector<double>>& distances) = 0;
};

/**
 * RSWrapper is a wrapper class for most RangeSearch types.
 */
template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
class RSWrapper : public RSWrapperBase
{
 public:
  //! Create the RSWrapper object.
  RSWrapper(const bool singleMode, const bool naive) :
      rs(singleMode, naive)
  {
    // Nothing else to do.
  }

  //! Create a new RSWrapper that is the same as this one.  This function
  //! will properly handle polymorphism.
  virtual RSWrapper* Clone() const { return new RSWrapper(*this); }

  //! Destruct the RSWrapper (nothing to do).
  virtual ~RSWrapper() { }

  //! Get the dataset.
  const arma::mat& Dataset() const { return rs.ReferenceSet(); }

  //! Get whether single-tree search is being used.
  bool SingleMode() const { return rs.SingleMode(); }
  //! Modify whether single-tree search is being used.
  bool& SingleMode() { return rs.SingleMode(); }

  //! Get whether naive search is being used.
  bool Naive() const { return rs.Naive(); }
  //! Modify whether naive search is being used.
  bool& Naive() { return rs.Naive(); }

  //! Train the model (build the reference tree if needed).  This ignores the
  //! leaf size.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t /* leafSize */);

  //! Perform bichromatic range search (i.e. a search with a separate query
  //! set).  This ignores the leaf size.
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const Range& range,
                      std::vector<std::vector<size_t>>& neighbors,
                      std::vector<std::vector<double>>& distances,
                      const size_t /* leafSize */);

  //! Perform monochromatic range search (i.e. a search with the reference set
  //! as the query set).
  virtual void Search(util::Timers& timers,
                      const Range& range,
                      std::vector<std::vector<size_t>>& neighbors,
                      std::vector<std::vector<double>>& distances);

  //! Serialize the RangeSearch model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(rs));
  }

 protected:
  using RSType = RangeSearch<EuclideanDistance, arma::mat, TreeType>;

  //! The instantiated RangeSearch object that we are wrapping.
  RSType rs;
};

/**
 * LeafSizeRSWrapper wraps any RangeSearch type that needs to be able to take
 * the leaf size into account when building trees.  The implementations of
 * Train() and bichromatic Search() take this leaf size into account.
 */
template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
class LeafSizeRSWrapper : public RSWrapper<TreeType>
{
 public:
  //! Construct the LeafSizeRSWrapper by delegating to the RSWrapper
  //! constructor.
  LeafSizeRSWrapper(const bool singleMode, const bool naive) :
      RSWrapper<TreeType>(singleMode, naive)
  {
    // Nothing else to do.
  }

  //! Delete the LeafSizeRSWrapper.
  virtual ~LeafSizeRSWrapper() { }

  //! Return a copy of the LeafSizeRSWrapper.
  virtual LeafSizeRSWrapper* Clone() const
  {
    return new LeafSizeRSWrapper(*this);
  }

  //! Train a model with the given parameters.  This overload uses leafSize.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize);

  //! Perform bichromatic search (e.g. search with a separate query set).  This
  //! overload takes the leaf size into account when building the query tree.
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const Range& range,
                      std::vector<std::vector<size_t>>& neighbors,
                      std::vector<std::vector<double>>& distances,
                      const size_t leafSize);

  //! Serialize the RangeSearch model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(rs));
  }

 protected:
  using RSWrapper<TreeType>::rs;
};

/**
 * The RSModel class provides an abstraction for the RangeSearch class,
 * abstracting away the TreeType parameter and allowing it to be specified at
 * runtime.  This class is written for the sake of the `range_search` binding,
 * but is not necessarily restricted to that usage.
 */
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
   * Copy the given RSModel.
   *
   * @param other RSModel to copy.
   */
  RSModel(const RSModel& other);

  /**
   * Take ownership of the given RSModel.
   *
   * @param other RSModel to take ownership of.
   */
  RSModel(RSModel&& other);

  /**
   * Copy the given RSModel.
   *
   * @param other RSModel to copy.
   */
  RSModel& operator=(const RSModel& other);

  /**
   * Take ownership of the given RSModel's data.
   *
   * @param other RSModel to copy.
   */
  RSModel& operator=(RSModel&& other);

  /**
   * Clean memory, if necessary.
   */
  ~RSModel();

  //! Serialize the range search model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Expose the dataset.
  const arma::mat& Dataset() const { return rSearch->Dataset(); }

  //! Get whether the model is in single-tree search mode.
  bool SingleMode() const { return rSearch->SingleMode(); }
  //! Modify whether the model is in single-tree search mode.
  bool& SingleMode() { return rSearch->SingleMode(); }

  //! Get whether the model is in naive search mode.
  bool Naive() const { return rSearch->Naive(); }
  //! Modify whether the model is in naive search mode.
  bool& Naive() { return rSearch->Naive(); }

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
   * Allocate the memory for the range search model.
   */
  void InitializeModel(const bool naive, const bool singleMode);

  /**
   * Build the reference tree on the given dataset with the given parameters.
   * This takes possession of the reference set to avoid a copy.
   *
   * @param referenceSet Set of reference points.
   * @param leafSize Leaf size of tree (ignored for the cover tree).
   * @param naive Whether naive search should be used.
   * @param singleMode Whether single-tree search should be used.
   */
  void BuildModel(util::Timers& timers,
                  arma::mat&& referenceSet,
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
  void Search(util::Timers& timers,
              arma::mat&& querySet,
              const Range& range,
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
  void Search(util::Timers& timers,
              const Range& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<double>>& distances);

 private:
  //! The type of tree we are using.
  TreeTypes treeType;
  //! (Only used for some tree types.)  The leaf size to use when building a
  //! tree.
  size_t leafSize;

  //! If true, we randomly project the data into a new basis before search.
  bool randomBasis;
  //! Random projection matrix.
  arma::mat q;

  /**
   * rSearch holds an instance of the RangeSearch class for the current
   * treeType. It is initialized every time BuildModel is executed.
   */
  RSWrapperBase* rSearch;

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

} // namespace mlpack

// Include implementation (of serialize() and templated wrapper classes).
#include "rs_model_impl.hpp"

#endif
