/**
 * @file rs_model.hpp
 * @author Ryan Curtin
 *
 * This is a model for range search.  It is useful in that it provides an easy
 * way to serialize a model, abstracts away the different types of trees, and
 * also reflects the RangeSearch API and automatically directs to the right
 * tree types.
 */
#ifndef MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_HPP
#define MLPACK_METHODS_RANGE_SEARCH_RS_MODEL_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>

#include "range_search.hpp"

namespace mlpack {
namespace range {

//! The mostly specified type of range search model.
template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
using RSType = RangeSearch<metric::EuclideanDistance, arma::mat, TreeType>;


/* MonoSearchVisitor executes a monochromatic range search on the given
 * RSType.(That is it does take the query set and the reference set the 
 * same
 */

class MonoSearchVisitor : public boost::static_visitor<void>
{
    private:
        //! range Range to search for.
        const math::Range& range;
        //! neighbors Output: neighbors falling within the desired range.
        std::vector<std::vector<size_t>>& neighbors;
        //! distances Output: distances of neighbors.
        std::vector<std::vector<double>>& distances;
    public:
        //! Perform monochromatic range search.
        template<typename RSType>
        void operator()(RSType* rs) const;

        //! Construct the MonoSearchVisitor object with the given parameters.
        MonoSearchVisitor(const math::Range& range,
                          std::vector<std::vector<size_t>>& neighbors,
                          std::vector<std::vector<double>>& distances) :
            range(range),
            neighbors(neighbors),
            distances(distances)
    {};
};


/**
   * BiSearchVisitor executes a bichromatic range search on the given RSType. 
   * This takes possession of the query set, so the query set will not be usable
   * after the search.
   */

class BiSearchVisitor : public boost::static_visitor<void>
{
 private:
   //! querySet Set of query points.
    arma::mat&& querySet;
   //! param range Range to search for.
    const math::Range& range;
   //! neighbors Output: neighbors falling within the desired range.
    std::vector<std::vector<size_t>>& neighbors;
   //! distances Output: distances of neighbors.
    std::vector<std::vector<double>>& distances;
    //! The no of points in a leaf (For binarySpacetrees)
    const size_t leafSize;
    //! Bichromatic range search on the given RSType considering the leafSize
    template<typename RSType>
    void SearchLeaf(RSType* rs) const;

 public:
    template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
    using RSTypeT = RSType<TreeType>;

    //! Default Bichromatic range search on the given RSType instance.    
    template<template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
    void operator()(RSTypeT<Treetype>* rs) const;
    
    //! Bichrmomatic range search specialized for KDTrees
    void operator()(RSTypeT<tree::KDTree>* rs) const;
    
    //! Bichromatic range search specialized for Ball Trees
    void operator()(RSTypeT<tree::BallTree>* rs) const;
    
    //! Construct the BiSearchVisitor.
    BiSearchVisitor(arma::mat& querySet,
                    const math::Range& range,
                    std::vector<std::vector<size_t>>& neighbors,
                    std::vector<std::vector<double>>& distances,
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
    const arma::mat& operator()(RSType *rs) const;
};


/**
 * SingleModeVisitor exposes whether the model is in single-tree search mode.
 */
class SingleModeVisitor : public boost::static_visitor<bool>
{
 public:
    //! Return whether in single-tree search mode
    template<typename RSType>
    bool operator()(RSType *rs) const;
};


/**
 * SetSingleModeVisitor lets modify whether the model is in single-tree
 * search mode
 */
class SetSingleModeVisitor : public boost::static_visitor<bool&>
{
    template<typename RSType>
    bool& operator()(RSType *rs);
};


/**
 * NaiveVisitor gets whether the model is in naive search mode.
 */
class NaiveVisitor : public boost::static_visitor<bool>
{
    template<typename RSType>
    bool operator()(RSType *rs) const;
};



/**
 * SetNaiveVisitor modifies whether the model is in naive search mode.
 */
class SetNaiveVisitor : public boost::static_visitor<bool&>
{
    template<typename RSType>
    bool operator()(RSType *rs);
};


/**
 * DeleteVisitor deletes the given RSType instance.
 */
class DeleteVisitor : public boost::static_visitor<void>
{
 public:
    //! Delete the RSType object.
    template<typename RSType>
    void operator()(RSType *rs) const;
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
    MAX_RP_TREE
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
   * treeType. 
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
                 RSType<tree::MaxRPTree>* rSearch;

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
