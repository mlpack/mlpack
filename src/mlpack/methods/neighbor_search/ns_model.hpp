/**
 * @file ns_model.hpp
 * @author Ryan Curtin
 *
 * This is a model for nearest or furthest neighbor search.  It is useful in
 * that it provides an easy way to serialize a model, abstracts away the
 * different types of trees, and also reflects the NeighborSearch API and
 * automatically directs to the right tree type.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_NS_MODEL_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
#include <mlpack/core/tree/spill_tree.hpp>
#include <boost/variant.hpp>
#include "neighbor_search.hpp"
#include "spill_search.hpp"

namespace mlpack {
namespace neighbor {

/**
 * Alias template for euclidean neighbor search.
 */
template<typename SortPolicy,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
using NSType = NeighborSearch<SortPolicy,
                              metric::EuclideanDistance,
                              arma::mat,
                              TreeType,
                              TreeType<metric::EuclideanDistance,
                                  NeighborSearchStat<SortPolicy>,
                                  arma::mat>::template DualTreeTraverser>;

/**
 * Alias template for euclidean spill search.
 */
using NSSpillType = SpillSearch<metric::EuclideanDistance, arma::mat>;

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

/**
 * MonoSearchVisitor executes a monochromatic neighbor search on the given
 * NSType. We don't make any difference for different instantiations of NSType.
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
  template<typename NSType>
  void operator()(NSType* ns) const;

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
 * BiSearchVisitor executes a bichromatic neighbor search on the given NSType.
 * We use template specialization to differenciate those tree types that
 * accept leafSize as a parameter. In these cases, before doing neighbor search,
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
  //! The result matrix for neighbors.
  arma::Mat<size_t>& neighbors;
  //! The result matrix for distances.
  arma::mat& distances;
  //! The number of points in a leaf (for BinarySpaceTrees).
  const size_t leafSize;
  //! Overlapping size (for spill trees).
  const double tau;

  //! Bichromatic neighbor search on the given NSType considering the leafSize.
  template<typename NSType>
  void SearchLeaf(NSType* ns) const;

 public:
  //! Alias template necessary for visual c++ compiler.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using NSTypeT = NSType<SortPolicy, TreeType>;

  //! Default Bichromatic neighbor search on the given NSType instance.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(NSTypeT<TreeType>* ns) const;

  //! Bichromatic neighbor search on the given NSType specialized for KDTrees.
  void operator()(NSTypeT<tree::KDTree>* ns) const;

  //! Bichromatic neighbor search on the given NSType specialized for BallTrees.
  void operator()(NSTypeT<tree::BallTree>* ns) const;

  //! Bichromatic neighbor search specialized for SPTrees.
  void operator()(NSSpillType* ns) const;

  //! Construct the BiSearchVisitor.
  BiSearchVisitor(const arma::mat& querySet,
                  const size_t k,
                  arma::Mat<size_t>& neighbors,
                  arma::mat& distances,
                  const size_t leafSize,
                  const double tau);
};

/**
 * TrainVisitor sets the reference set to a new reference set on the given
 * NSType. We use template specialization to differenciate those tree types that
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
  //! Overlapping size (for spill trees).
  const double tau;

  //! Train on the given NSType considering the leafSize.
  template<typename NSType>
  void TrainLeaf(NSType* ns) const;

 public:
  //! Alias template necessary for visual c++ compiler.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using NSTypeT = NSType<SortPolicy, TreeType>;

  //! Default Train on the given NSType instance.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(NSTypeT<TreeType>* ns) const;

  //! Train on the given NSType specialized for KDTrees.
  void operator()(NSTypeT<tree::KDTree>* ns) const;

  //! Train on the given NSType specialized for BallTrees.
  void operator()(NSTypeT<tree::BallTree>* ns) const;

  //! Train specialized for SPTrees.
  void operator()(NSSpillType* ns) const;

  //! Construct the TrainVisitor object with the given reference set, leafSize
  //! for BinarySpaceTrees, and tau for spill trees.
  TrainVisitor(arma::mat&& referenceSet,
               const size_t leafSize,
               const double tau);
};

/**
 * SingleModeVisitor exposes the SingleMode method of the given NSType.
 */
class SingleModeVisitor : public boost::static_visitor<bool&>
{
 public:
  //! Return whether or not single-tree search is enabled.
  template<typename NSType>
  bool& operator()(NSType* ns) const;
};

/**
 * NaiveVisitor exposes the Naive method of the given NSType.
 */
class NaiveVisitor : public boost::static_visitor<bool&>
{
 public:
  //! Return whether or not naive search is enabled.
  template<typename NSType>
  bool& operator()(NSType *ns) const;
};

/**
 * EpsilonVisitor exposes the Epsilon method of the given NSType.
 */
class EpsilonVisitor : public boost::static_visitor<double&>
{
 public:
  //! Return epsilon, the approximation parameter.
  template<typename NSType>
  double& operator()(NSType *ns) const;
};

/**
 * ReferenceSetVisitor exposes the referenceSet of the given NSType.
 */
class ReferenceSetVisitor : public boost::static_visitor<const arma::mat&>
{
 public:
  //! Return the reference set.
  template<typename NSType>
  const arma::mat& operator()(NSType *ns) const;
};

/**
 * DeleteVisitor deletes the given NSType instance.
 */
class DeleteVisitor : public boost::static_visitor<void>
{
 public:
  //! Delete the NSType object.
  template<typename NSType>
  void operator()(NSType *ns) const;
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
    SPILL_TREE
  };

 private:
  //! Tree type considered for neighbor search.
  TreeTypes treeType;

  //! For tree types that accept the maxLeafSize parameter.
  size_t leafSize;

  //! Overlapping size (for spill trees).
  double tau;

  //! If true, random projections are used.
  bool randomBasis;
  //! This is the random projection matrix; only used if randomBasis is true.
  arma::mat q;

  /**
   * nSearch holds an instance of the NeigborSearch class for the current
   * treeType. It is initialized every time BuildModel is executed.
   * We access to the contained value through the visitor classes defined above.
   */
  boost::variant<NSType<SortPolicy, tree::KDTree>*,
                 NSType<SortPolicy, tree::StandardCoverTree>*,
                 NSType<SortPolicy, tree::RTree>*,
                 NSType<SortPolicy, tree::RStarTree>*,
                 NSType<SortPolicy, tree::BallTree>*,
                 NSType<SortPolicy, tree::XTree>*,
                 NSType<SortPolicy, tree::HilbertRTree>*,
                 NSType<SortPolicy, tree::RPlusTree>*,
                 NSType<SortPolicy, tree::RPlusPlusTree>*,
                 NSSpillType*> nSearch;

 public:
  /**
   * Initialize the NSModel with the given type and whether or not a random
   * basis should be used.
   */
  NSModel(TreeTypes treeType = TreeTypes::KD_TREE, bool randomBasis = false);

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

  //! Expose naiveMode.
  bool Naive() const;
  bool& Naive();

  //! Expose Epsilon.
  double Epsilon() const;
  double& Epsilon();

  //! Expose leafSize.
  size_t LeafSize() const { return leafSize; }
  size_t& LeafSize() { return leafSize; }

  //! Expose tau.
  double Tau() const { return tau; }
  double& Tau() { return tau; }

  //! Expose treeType.
  TreeTypes TreeType() const { return treeType; }
  TreeTypes& TreeType() { return treeType; }

  //! Expose randomBasis.
  bool RandomBasis() const { return randomBasis; }
  bool& RandomBasis() { return randomBasis; }

  //! Build the reference tree.
  void BuildModel(arma::mat&& referenceSet,
                  const size_t leafSize,
                  const bool naive,
                  const bool singleMode,
                  const double epsilon = 0);

  //! Perform neighbor search.  The query set will be reordered.
  void Search(arma::mat&& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Perform monochromatic neighbor search.
  void Search(const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  //! Return a string representation of the current tree type.
  std::string TreeName() const;
};

} // namespace neighbor
} // namespace mlpack

// Include implementation.
#include "ns_model_impl.hpp"

#endif
