/**
 * @file methods/neighbor_search/ns_model.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also (roughly) reflects the NeighborSearch API
 * and automatically directs to the right tree type.  It is meant to be used by
 * the knn and kfn bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
#include <mlpack/core/tree/spill_tree.hpp>
#include <mlpack/core/tree/octree.hpp>
#include "neighbor_search.hpp"

namespace mlpack {

/**
 * NSWrapperBase is a base wrapper class for holding all NeighborSearch types
 * supported by NSModel.  All NeighborSearch type wrappers inherit from this
 * class, allowing a simple interface via inheritance for all the different
 * types we want to support.
 */
class NSWrapperBase
{
 public:
  //! Create the NSWrapperBase object.  The base class does not hold anything,
  //! so this constructor does not do anything.
  NSWrapperBase() { }

  //! Create a new NSWrapperBase that is the same as this one.  This function
  //! will properly handle polymorphism.
  virtual NSWrapperBase* Clone() const = 0;

  //! Destruct the NSWrapperBase (nothing to do).
  virtual ~NSWrapperBase() { }

  //! Return a reference to the dataset.
  virtual const arma::mat& Dataset() const = 0;

  //! Get the search mode.
  virtual NeighborSearchMode SearchMode() const = 0;
  //! Modify the search modem
  virtual NeighborSearchMode& SearchMode() = 0;

  //! Get the approximation parameter epsilon.
  virtual double Epsilon() const = 0;
  //! Modify the approximation parameter epsilon.
  virtual double& Epsilon() = 0;

  //! Train the NeighborSearch model with the given parameters.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize,
                     const double tau,
                     const double rho) = 0;

  //! Perform bichromatic neighbor search (i.e. search with a separate query
  //! set).
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      const size_t leafSize,
                      const double rho) = 0;

  //! Perform monochromatic neighbor search (i.e. use the reference set as the
  //! query set).
  virtual void Search(util::Timers& timers,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances) = 0;
};

/**
 * NSWrapper is a wrapper class for most NeighborSearch types.
 */
template<typename SortPolicy,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class DualTreeTraversalType =
             TreeType<EuclideanDistance,
                      NeighborSearchStat<SortPolicy>,
                      arma::mat>::template DualTreeTraverser,
         template<typename RuleType> class SingleTreeTraversalType =
             TreeType<EuclideanDistance,
                      NeighborSearchStat<SortPolicy>,
                      arma::mat>::template SingleTreeTraverser>
class NSWrapper : public NSWrapperBase
{
 public:
  //! Construct the NSWrapper object, initializing the internally-held
  //! NeighborSearch object.
  NSWrapper(const NeighborSearchMode searchMode,
            const double epsilon) :
      ns(searchMode, epsilon)
  {
    // Nothing else to do.
  }

  //! Delete the NSWrapper object.
  virtual ~NSWrapper() { }

  //! Create a copy of this NSWrapper object.  This correctly handles
  //! polymorphism.
  virtual NSWrapper* Clone() const { return new NSWrapper(*this); }

  //! Get a reference to the reference set.
  const arma::mat& Dataset() const { return ns.ReferenceSet(); }

  //! Get the search mode.
  NeighborSearchMode SearchMode() const { return ns.SearchMode(); }
  //! Modify the search mode.
  NeighborSearchMode& SearchMode() { return ns.SearchMode(); }

  //! Get epsilon, the approximation parameter.
  double Epsilon() const { return ns.Epsilon(); }
  //! Modify epsilon, the approximation parameter.
  double& Epsilon() { return ns.Epsilon(); }

  //! Train the model with the given options.  For NSWrapper, we ignore the
  //! extra parameters.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t /* leafSize */,
                     const double /* tau */,
                     const double /* rho */);

  //! Perform bichromatic neighbor search (i.e. search with a separate query
  //! set).  For NSWrapper, we ignore the extra parameters.
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      const size_t /* leafSize */,
                      const double /* rho */);

  //! Perform monochromatic neighbor search (i.e. use the reference set as the
  //! query set).
  virtual void Search(util::Timers& timers,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances);

  //! Serialize the NeighborSearch model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(ns));
  }

 protected:
  // Convenience typedef for the neighbor search type held by this class.
  using NSType = NeighborSearch<SortPolicy,
                                EuclideanDistance,
                                arma::mat,
                                TreeType,
                                DualTreeTraversalType,
                                SingleTreeTraversalType>;

  //! The instantiated NeighborSearch object that we are wrapping.
  NSType ns;
};

/**
 * LeafSizeNSWrapper wraps any NeighborSearch types that take a leaf size for
 * tree construction.  The implementations of Train() and Search() take the leaf
 * size into account.
 */
template<typename SortPolicy,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename RuleType> class DualTreeTraversalType =
             TreeType<EuclideanDistance,
                      NeighborSearchStat<SortPolicy>,
                      arma::mat>::template DualTreeTraverser,
         template<typename RuleType> class SingleTreeTraversalType =
             TreeType<EuclideanDistance,
                      NeighborSearchStat<SortPolicy>,
                      arma::mat>::template SingleTreeTraverser>
class LeafSizeNSWrapper :
    public NSWrapper<SortPolicy,
                     TreeType,
                     DualTreeTraversalType,
                     SingleTreeTraversalType>
{
 public:
  //! Construct the LeafSizeNSWrapper by delegating to the NSWrapper
  //! constructor.
  LeafSizeNSWrapper(const NeighborSearchMode searchMode,
                    const double epsilon) :
      NSWrapper<SortPolicy,
                TreeType,
                DualTreeTraversalType,
                SingleTreeTraversalType>(searchMode, epsilon)
  {
    // Nothing to do.
  }

  //! Delete the LeafSizeNSWrapper.
  virtual ~LeafSizeNSWrapper() { }

  //! Return a copy of the LeafSizeNSWrapper.
  virtual LeafSizeNSWrapper* Clone() const
  {
    return new LeafSizeNSWrapper(*this);
  }

  //! Train a model with the given parameters.  This overload uses leafSize but
  //! ignores the other parameters.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize,
                     const double /* tau */,
                     const double /* rho */);

  //! Perform bichromatic search (e.g. search with a separate query set).  This
  //! overload uses the leaf size, but ignores the other parameters.
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      const size_t leafSize,
                      const double /* rho */);

  //! Serialize the NeighborSearch model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(ns));
  }

 protected:
  using NSWrapper<SortPolicy,
                  TreeType,
                  DualTreeTraversalType,
                  SingleTreeTraversalType>::ns;
};

/**
 * The SpillNSWrapper class wraps the NeighborSearch class when the spill tree
 * is used.
 */
template<typename SortPolicy>
class SpillNSWrapper :
    public NSWrapper<
        SortPolicy,
        SPTree,
        SPTree<EuclideanDistance,
               NeighborSearchStat<SortPolicy>,
               arma::mat>::template DefeatistDualTreeTraverser,
        SPTree<EuclideanDistance,
               NeighborSearchStat<SortPolicy>,
               arma::mat>::template DefeatistSingleTreeTraverser>
{
 public:
  //! Construct the SpillNSWrapper.
  SpillNSWrapper(const NeighborSearchMode searchMode,
                 const double epsilon) :
      NSWrapper<
          SortPolicy,
          SPTree,
          SPTree<EuclideanDistance,
                 NeighborSearchStat<SortPolicy>,
                 arma::mat>::template DefeatistDualTreeTraverser,
          SPTree<EuclideanDistance,
                 NeighborSearchStat<SortPolicy>,
                 arma::mat>::template DefeatistSingleTreeTraverser>(
          searchMode, epsilon)
  {
    // Nothing to do.
  }

  //! Destruct the SpillNSWrapper.
  virtual ~SpillNSWrapper() { }

  //! Return a copy of the SpillNSWrapper.
  virtual SpillNSWrapper* Clone() const { return new SpillNSWrapper(*this); }

  //! Train the model using the given parameters.
  virtual void Train(util::Timers& timers,
                     arma::mat&& referenceSet,
                     const size_t leafSize,
                     const double tau,
                     const double rho);

  //! Perform bichromatic search (i.e. search with a different query set) using
  //! the given parameters.
  virtual void Search(util::Timers& timers,
                      arma::mat&& querySet,
                      const size_t k,
                      arma::Mat<size_t>& neighbors,
                      arma::mat& distances,
                      const size_t leafSize,
                      const double rho);

  //! Serialize the NeighborSearch model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(ns));
  }

 protected:
  using NSWrapper<
      SortPolicy,
      SPTree,
      SPTree<EuclideanDistance,
             NeighborSearchStat<SortPolicy>,
             arma::mat>::template DefeatistDualTreeTraverser,
      SPTree<EuclideanDistance,
             NeighborSearchStat<SortPolicy>,
             arma::mat>::template DefeatistSingleTreeTraverser>::ns;
};

/**
 * The NSModel class provides an easy way to serialize a model, abstracts away
 * the different types of trees, and also reflects the NeighborSearch API.  This
 * class is meant to be used by the command-line mlpack_knn and mlpack_kfn
 * programs, and thus does not have the same complete functionality and
 * flexibility as the NeighborSearch class.  So if you are using it outside of
 * mlpack_knn and mlpack_kfn, be aware that it is limited!
 *
 * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
 */
template<typename SortPolicy>
class NSModel
{
 public:
  //! Enum type to identify each accepted tree type.
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
    SPILL_TREE,
    UB_TREE,
    OCTREE
  };

 private:
  //! Tree type considered for neighbor search.
  TreeTypes treeType;

  //! If true, random projections are used.
  bool randomBasis;
  //! This is the random projection matrix; only used if randomBasis is true.
  arma::mat q;

  size_t leafSize;
  double tau;
  double rho;

  /**
   * nSearch holds an instance of the NeighborSearch class for the current
   * treeType. It is initialized every time BuildModel is executed.
   */
  NSWrapperBase* nSearch;

 public:
  /**
   * Initialize the NSModel with the given type and whether or not a random
   * basis should be used.
   *
   * @param treeType Type of tree to use.
   * @param randomBasis Whether or not to project the points onto a random basis
   *      before searching.
   */
  NSModel(TreeTypes treeType = TreeTypes::KD_TREE, bool randomBasis = false);

  /**
   * Copy the given NSModel.
   *
   * @param other Model to copy.
   */
  NSModel(const NSModel& other);

  /**
   * Take ownership of the given NSModel.
   *
   * @param other Model to take ownership of.
   */
  NSModel(NSModel&& other);

  /**
   * Copy the given NSModel.
   *
   * @param other Model to copy.
   */
  NSModel& operator=(const NSModel& other);

  /**
   * Take ownership of the given NSModel.
   *
   * @param other Model to take ownership of.
   */
  NSModel& operator=(NSModel&& other);

  //! Clean memory, if necessary.
  ~NSModel();

  //! Serialize the neighbor search model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

  //! Expose the dataset.
  const arma::mat& Dataset() const;

  //! Expose SearchMode.
  NeighborSearchMode SearchMode() const;
  NeighborSearchMode& SearchMode();

  //! Expose LeafSize.
  size_t LeafSize() const { return leafSize; }
  size_t& LeafSize() { return leafSize; }

  //! Expose Tau.
  double Tau() const { return tau; }
  double& Tau() { return tau; }

  //! Expose Rho.
  double Rho() const { return rho; }
  double& Rho() { return rho; }

  //! Expose Epsilon.
  double Epsilon() const;
  double& Epsilon();

  //! Expose treeType.
  TreeTypes TreeType() const { return treeType; }
  TreeTypes& TreeType() { return treeType; }

  //! Expose randomBasis.
  bool RandomBasis() const { return randomBasis; }
  bool& RandomBasis() { return randomBasis; }

  //! Initialize the model type.  (This does not perform any training.)
  void InitializeModel(const NeighborSearchMode searchMode,
                       const double epsilon);

  //! Build the reference tree.
  void BuildModel(util::Timers& timers,
                  arma::mat&& referenceSet,
                  const NeighborSearchMode searchMode,
                  const double epsilon = 0);

  //! Perform neighbor search.  The query set will be reordered.
  void Search(util::Timers& timers,
              arma::mat&& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Perform monochromatic neighbor search.
  void Search(util::Timers& timers,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Return a string representation of the current tree type.
  std::string TreeName() const;
};

} // namespace mlpack

// Include implementation.
#include "ns_model_impl.hpp"

#endif
