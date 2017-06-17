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
#include <boost/variant.hpp>
#include "range_search.hpp"

namespace mlpack {
namespace range {

/**
 * Alias template for Range Search.
 */
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
using RSType = RangeSearch<metric::EuclideanDistance, arma::mat, TreeType>;

struct RSModelName
{
  static const std::string Name() { return "range_search_model"; }
};

/**
 * MonoSearchVisitor executes a monochromatic range search on the given
 * RSType. Range Search is performed on the reference set itself, no querySet.
 */
class MonoSearchVisitor : public  boost::static_visitor<void>
{
 private:
  //! The range to search for.
  const math::Range& range;
  //! Output neighbors.
  std::vector<std::vector<size_t>>& neighbors;
  //! Output distances.
  std::vector<std::vector<double>>& distances;

 public:
  //! Perform monochromatic search with the given RangeSearch object.
  template<typename RSType>
  void operator()(RSType* rs) const;

  //! Construct the MonoSearchVisitor with the given parameters.
  MonoSearchVisitor(const math::Range& range,
                    std::vector<std::vector<size_t>>& neighbors,
                    std::vector<std::vector<double>>& distances):
      range(range),
      neighbors(neighbors),
      distances(distances)
  {};
};

/**
 * BiSearchVisitor executes a bichromatic range search on the given RSType.
 * We use template specialization to differentiate those tree types that
 * accept leafSize as a parameter. In these cases, before doing range search,
 * a query tree with proper leafSize is built from the querySet.
 */
class BiSearchVisitor : public boost::static_visitor<void>
{
 private:
  //! The query set for the bichromatic search.
  const arma::mat& querySet;
  //! Range to search neighbours for.
  const math::Range& range;
  //! The result vector for neighbors.
  std::vector<std::vector<size_t>>& neighbors;
  //! The result vector for distances.
  std::vector<std::vector<double>>& distances;
  //! The number of points in a leaf (for BinarySpaceTrees).
  const size_t leafSize;

  //! Bichromatic range search on the given RSType considering the leafSize.
  template<typename RSType>
  void SearchLeaf(RSType* rs) const;

 public:
  //! Alias template necessary for visual c++ compiler.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using RSTypeT = RSType<TreeType>;

  //! Default Bichromatic range search on the given RSType instance.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(RSTypeT<TreeType>* rs) const;

  //! Bichromatic range search on the given RSType specialized for KDTrees.
  void operator()(RSTypeT<tree::KDTree>* rs) const;

  //! Bichromatic range search on the given RSType specialized for BallTrees.
  void operator()(RSTypeT<tree::BallTree>* rs) const;

  //! Bichromatic range search specialized for octrees.
  void operator()(RSTypeT<tree::Octree>* rs) const;

  //! Construct the BiSearchVisitor.
  BiSearchVisitor(const arma::mat& querySet,
                  const math::Range& range,
                  std::vector<std::vector<size_t>>& neighbors,
                  std::vector<std::vector<double>>& distances,
                  const size_t leafSize);
};

/**
 * TrainVisitor sets the reference set to a new reference set on the given
 * RSType. We use template specialization to differentiate those tree types that
 * accept leafSize as a parameter. In these cases, a reference tree with proper
 * leafSize is built from the referenceSet.
 */
class TrainVisitor : public boost::static_visitor<void>
{
 private:
  //! The reference set to use for training.
  arma::mat&& referenceSet;
  //! The leaf size, used only by BinarySpaceTree.
  size_t leafSize;
  //! Train on the given RsType considering the leafSize.
  template<typename RSType>
  void TrainLeaf(RSType* rs) const;

 public:
  //! Alias template necessary for visual c++ compiler.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  using RSTypeT = RSType<TreeType>;

  //! Default Train on the given RSType instance.
  template<template<typename TreeMetricType,
                    typename TreeStatType,
                    typename TreeMatType> class TreeType>
  void operator()(RSTypeT<TreeType>* rs) const;

  //! Train on the given RSType specialized for KDTrees.
  void operator()(RSTypeT<tree::KDTree>* rs) const;

  //! Train on the given RSType specialized for BallTrees.
  void operator()(RSTypeT<tree::BallTree>* rs) const;

  //! Train specialized for octrees.
  void operator()(RSTypeT<tree::Octree>* rs) const;

  //! Construct the TrainVisitor object with the given reference set, leafSize
  TrainVisitor(arma::mat&& referenceSet,
               const size_t leafSize);
};

/**
 * ReferenceSetVisitor exposes the referenceSet of the given RSType.
 */
class ReferenceSetVisitor : public boost::static_visitor<const arma::mat&>
{
 public:
  //! Return the reference set.
  template<typename RSType>
  const arma::mat& operator()(RSType* rs) const;
};

/**
 * DeleteVisitor deletes the given RSType instance.
 */
class DeleteVisitor : public boost::static_visitor<void>
{
 public:
  //! Delete the RSType object.
  template<typename RSType>
  void operator()(RSType* rs) const;
};

/**
 * Exposes the seralize method of the given RSType.
 */
template<typename Archive>
class SerializeVisitor : public boost::static_visitor<void>
{
 private:
  //! Archive to serialize to.
  Archive& ar;
  //! Name of the model to serialize.
  const std::string& name;

 public:
  //! Serialize the given model.
  template<typename RSType>
  void operator()(RSType* rs) const;

  //! Construct the SerializeVisitor with the given archive and name.
  SerializeVisitor(Archive& ar, const std::string& name);
};

/**
 * SingleModeVisitor exposes the SingleMode() method of the given RSType.
 */
class SingleModeVisitor : public boost::static_visitor<bool&>
{
 public:
  /**
   * Get a reference to the singleMode parameter of the given RangeSeach
   * object.
   */
  template<typename RSType>
  bool& operator()(RSType* rs) const;
};

/**
 * NaiveVisitor exposes the Naive() method of the given RSType.
 */
class NaiveVisitor : public boost::static_visitor<bool&>
{
 public:
  /**
   * Get a reference to the naive parameter of the given RangeSearch object.
   */
  template<typename RSType>
  bool& operator()(RSType* rs) const;
};

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

  /**
   * rSearch holds an instance of the RangeSearch class for the current
   * treeType. It is initialized every time BuildModel is executed.
   * We access to the contained value through the visitor classes defined above.
   */
  boost::variant<RSType<tree::KDTree>*,
                 RSType<tree::StandardCoverTree>*,
                 RSType<tree::RTree>*,
                 RSType<tree::RStarTree>*,
                 RSType<tree::BallTree>*,
                 RSType<tree::XTree>*,
                 RSType<tree::HilbertRTree>*,
                 RSType<tree::RPlusTree>*,
                 RSType<tree::RPlusPlusTree>*,
                 RSType<tree::VPTree>*,
                 RSType<tree::RPTree>*,
                 RSType<tree::MaxRPTree>*,
                 RSType<tree::UBTree>*,
                 RSType<tree::Octree>*> rSearch;

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
   * Take ownership of the given RSModel.
   *
   * @param other RSModel to take ownership of.
   */
  RSModel& operator=(RSModel&& other);

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
