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
#include <boost/variant.hpp>
#include "neighbor_search.hpp"

namespace mlpack {
namespace neighbor {

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

class MonoSearchVisitor : public boost::static_visitor<void>
{
 private:
   const size_t k;
   arma::Mat<size_t>& neighbors;
   arma::mat& distances;

 public:
   template<typename NSType>
   void operator()(NSType *ns) const;

   MonoSearchVisitor(const size_t k,
                     arma::Mat<size_t>& neighbors,
                     arma::mat& distances);
};

class BiSearchVisitor : public boost::static_visitor<void>
{
 private:
   const arma::mat& querySet;
   const size_t k;
   arma::Mat<size_t>& neighbors;
   arma::mat& distances;
   const size_t leafSize;

   template<typename NSType>
   void SearchLeaf(NSType *ns) const;

 public:
   template<typename SortPolicy,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   void operator()(NSType<SortPolicy,TreeType> *ns) const;

   template<typename SortPolicy>
   void operator()(NSType<SortPolicy,tree::KDTree> *ns) const;

   template<typename SortPolicy>
   void operator()(NSType<SortPolicy,tree::BallTree> *ns) const;

   BiSearchVisitor(const arma::mat& querySet,
                   const size_t k,
                   arma::Mat<size_t>& neighbors,
                   arma::mat& distances,
                   const size_t leafSize);
};

class TrainVisitor : public boost::static_visitor<void>
{
 private:
   arma::mat&& referenceSet;
   size_t leafSize;

   template<typename NSType>
   void TrainLeaf(NSType* ns) const;

 public:
   template<typename SortPolicy,
            template<typename TreeMetricType,
                     typename TreeStatType,
                     typename TreeMatType> class TreeType>
   void operator()(NSType<SortPolicy,TreeType> *ns) const;

   template<typename SortPolicy>
   void operator()(NSType<SortPolicy,tree::KDTree> *ns) const;

   template<typename SortPolicy>
   void operator()(NSType<SortPolicy,tree::BallTree> *ns) const;

   TrainVisitor(arma::mat&& referenceSet, const size_t leafSize);
};

class SingleModeVisitor : public boost::static_visitor<bool&>
{
 public:
   template<typename NSType>
   bool& operator()(NSType *ns) const;
};

class NaiveVisitor : public boost::static_visitor<bool&>
{
 public:
   template<typename NSType>
   bool& operator()(NSType *ns) const;
};

class ReferenceSetVisitor : public boost::static_visitor<const arma::mat&>
{
 public:
   template<typename NSType>
   const arma::mat& operator()(NSType *ns) const;
};

class DeleteVisitor : public boost::static_visitor<void>
{
 public:
   template<typename NSType>
   void operator()(NSType *ns) const;
};

template<typename Archive>
class SerializeVisitor : public boost::static_visitor<void>
{
 private:
   Archive& ar;
   const std::string& name;

 public:
   template<typename NSType>
   void operator()(NSType *ns) const;

   SerializeVisitor(Archive& ar, const std::string& name);
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
    BALL_TREE,
    X_TREE
  };

 private:
  TreeTypes treeType;
  size_t leafSize;

  // For random projections.
  bool randomBasis;
  arma::mat q;

  boost::variant<NSType<SortPolicy, tree::KDTree>*,
                 NSType<SortPolicy, tree::StandardCoverTree>*,
                 NSType<SortPolicy, tree::RTree>*,
                 NSType<SortPolicy, tree::RStarTree>*,
                 NSType<SortPolicy, tree::BallTree>*,
                 NSType<SortPolicy, tree::XTree>*> nSearch;

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

  bool Naive() const;
  bool& Naive();

  size_t LeafSize() const { return leafSize; }
  size_t& LeafSize() { return leafSize; }

  TreeTypes TreeType() const { return treeType; }
  TreeTypes& TreeType() { return treeType; }

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
