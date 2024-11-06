/**
 * @file methods/rann/ra_search.hpp
 * @author Parikshit Ram
 *
 * Defines the RASearch class, which performs an abstract rank-approximate
 * nearest/farthest neighbor query on two datasets.
 *
 * The details of this method can be found in the following paper:
 *
 * @code
 * @inproceedings{ram2009rank,
 *   title={{Rank-Approximate Nearest Neighbor Search: Retaining Meaning and
 *       Speed in High Dimensions}},
 *   author={{Ram, P. and Lee, D. and Ouyang, H. and Gray, A. G.}},
 *   booktitle={{Advances of Neural Information Processing Systems}},
 *   year={2009}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_SEARCH_HPP
#define MLPACK_METHODS_RANN_RA_SEARCH_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>

#include "ra_query_stat.hpp"
#include "ra_util.hpp"

namespace mlpack {

// Forward declaration.
template<template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
class LeafSizeRAWrapper;

/**
 * The RASearch class: This class provides a generic manner to perform
 * rank-approximate search via random-sampling. If the 'naive' option is chosen,
 * this rank-approximate search will be done by randomly sampling from the whole
 * set. If the 'naive' option is not chosen, the sampling is done in a
 * stratified manner in the tree as mentioned in the algorithms in Figure 2 of
 * the following paper:
 *
 * @code
 * @inproceedings{ram2009rank,
 *   title={{Rank-Approximate Nearest Neighbor Search: Retaining Meaning and
 *       Speed in High Dimensions}},
 *   author={{Ram, P. and Lee, D. and Ouyang, H. and Gray, A. G.}},
 *   booktitle={{Advances of Neural Information Processing Systems}},
 *   year={2009}
 * }
 * @endcode
 *
 * RASearch is currently known to not work with ball trees (#356).
 *
 * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
 * @tparam DistanceType The distance metric to use for computation.
 * @tparam TreeType The tree type to use.
 */
template<typename SortPolicy = NearestNeighborSort,
         typename DistanceType = EuclideanDistance,
         typename MatType = arma::mat,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType = KDTree>
class RASearch
{
 public:
  //! Convenience typedef.
  using Tree = TreeType<DistanceType, RAQueryStat<SortPolicy>, MatType>;

  /**
   * Initialize the RASearch object, passing both a reference dataset (this is
   * the dataset that will be searched).  Optionally, perform the computation in
   * naive mode or single-tree mode.  An initialized distance metric can be
   * given, for cases where the distance metric has internal data (i.e. the
   * distance::MahalanobisDistance class).
   *
   * This method will copy the matrices to internal copies, which are rearranged
   * during tree-building.  If you don't need to keep the reference dataset,
   * you can use std::move() to remove the overhead of making copies. Using
   * std::move() transfers the ownership of the dataset.
   *
   * tau, the rank-approximation parameter, specifies that we are looking for k
   * neighbors with probability alpha of being in the top tau percent of nearest
   * neighbors.  So, as an example, if our dataset has 1000 points, and we want
   * 5 nearest neighbors with 95% probability of being in the top 5% of nearest
   * neighbors (or, the top 50 nearest neighbors), we set k = 5, tau = 5, and
   * alpha = 0.95.
   *
   * The method will fail (and throw a std::invalid_argument exception) if the
   * value of tau is too low: tau must be set such that the number of points in
   * the corresponding percentile of the data is greater than k.  Thus, if we
   * choose tau = 0.1 with a dataset of 1000 points and k = 5, then we are
   * attempting to choose 5 nearest neighbors out of the closest 1 point -- this
   * is invalid.
   *
   * @param referenceSet Set of reference points.
   * @param naive If true, the rank-approximate search will be performed by
   *      directly sampling the whole set instead of using the stratified
   *      sampling on the tree.
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).  This is useful when Search() will be called with
   *      few query points.
   * @param distance An optional instance of the DistanceType class.
   * @param tau The rank-approximation in percentile of the data. The default
   *     value is 5%.
   * @param alpha The desired success probability. The default value is 0.95.
   * @param sampleAtLeaves Sample at leaves for faster but less accurate
   *      computation. This defaults to 'false'.
   * @param firstLeafExact Traverse to the first leaf without approximation.
   *     This can ensure that the query definitely finds its (near) duplicate
   *     if there exists one.  This defaults to 'false' for now.
   * @param singleSampleLimit The limit on the largest node that can be
   *     approximated by sampling. This defaults to 20.
   */
  RASearch(MatType referenceSet,
           const bool naive = false,
           const bool singleMode = false,
           const double tau = 5,
           const double alpha = 0.95,
           const bool sampleAtLeaves = false,
           const bool firstLeafExact = false,
           const size_t singleSampleLimit = 20,
           const DistanceType distance = DistanceType());

  /**
   * Initialize the RASearch object with the given pre-constructed reference
   * tree.  It is assumed that the points in the tree's dataset correspond to
   * the reference set.  Optionally, choose to use single-tree mode.  Naive mode
   * is not available as an option for this constructor; instead, to run naive
   * computation, use a different constructor.  Additionally, an instantiated
   * distance metric can be given, for cases where the distance metric holds
   * data.
   *
   * There is no copying of the data matrices in this constructor (because
   * tree-building is not necessary), so this is the constructor to use when
   * copies absolutely must be avoided.
   *
   * tau, the rank-approximation parameter, specifies that we are looking for k
   * neighbors with probability alpha of being in the top tau percent of nearest
   * neighbors.  So, as an example, if our dataset has 1000 points, and we want
   * 5 nearest neighbors with 95% probability of being in the top 5% of nearest
   * neighbors (or, the top 50 nearest neighbors), we set k = 5, tau = 5, and
   * alpha = 0.95.
   *
   * The method will fail (and throw a std::invalid_argument exception) if the
   * value of tau is too low: tau must be set such that the number of points in
   * the corresponding percentile of the data is greater than k.  Thus, if we
   * choose tau = 0.1 with a dataset of 1000 points and k = 5, then we are
   * attempting to choose 5 nearest neighbors out of the closest 1 point -- this
   * is invalid.
   *
   * @note
   * Tree-building may (at least with BinarySpaceTree) modify the ordering
   * of a matrix, so be aware that the results you get from Search() will
   * correspond to the modified matrix.
   *
   * @param referenceTree Pre-built tree for reference points.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param tau The rank-approximation in percentile of the data. The default
   *     value is 5%.
   * @param alpha The desired success probability. The default value is 0.95.
   * @param sampleAtLeaves Sample at leaves for faster but less accurate
   *      computation. This defaults to 'false'.
   * @param firstLeafExact Traverse to the first leaf without approximation.
   *     This can ensure that the query definitely finds its (near) duplicate
   *     if there exists one.  This defaults to 'false' for now.
   * @param singleSampleLimit The limit on the largest node that can be
   *     approximated by sampling. This defaults to 20.
   * @param distance Instantiated distance metric.
   */
  RASearch(Tree* referenceTree,
           const bool singleMode = false,
           const double tau = 5,
           const double alpha = 0.95,
           const bool sampleAtLeaves = false,
           const bool firstLeafExact = false,
           const size_t singleSampleLimit = 20,
           const DistanceType distance = DistanceType());

  /**
   * Create an RASearch object with no reference data.  If Search() is called
   * before a reference set is set with Train(), an exception will be thrown.
   *
   * @param naive Whether naive (brute-force) search should be used.
   * @param singleMode Whether single-tree computation should be used (as
   *      opposed to dual-tree computation).
   * @param tau The rank-approximation in percentile of the data. The default
   *     value is 5%.
   * @param alpha The desired success probability. The default value is 0.95.
   * @param sampleAtLeaves Sample at leaves for faster but less accurate
   *      computation. This defaults to 'false'.
   * @param firstLeafExact Traverse to the first leaf without approximation.
   *     This can ensure that the query definitely finds its (near) duplicate
   *     if there exists one.  This defaults to 'false' for now.
   * @param singleSampleLimit The limit on the largest node that can be
   *     approximated by sampling. This defaults to 20.
   * @param distance Instantiated distance metric.
   */
  RASearch(const bool naive = false,
           const bool singleMode = false,
           const double tau = 5,
           const double alpha = 0.95,
           const bool sampleAtLeaves = false,
           const bool firstLeafExact = false,
           const size_t singleSampleLimit = 20,
           const DistanceType distance = DistanceType());

  /**
   * Delete the RASearch object. The tree is the only member we are
   * responsible for deleting.  The others will take care of themselves.
   */
  ~RASearch();

  /**
   * "Train" the model on the given reference set.  If tree-based search is
   * being used (if Naive() is false), the reference tree is rebuilt. Thus, a
   * copy of the reference dataset is made. If you don't need to keep the
   * dataset, you can avoid copying by using std::move(). This transfers the
   * ownership of the dataset.
   *
   * @param referenceSet New reference set to use.
   */
  void Train(MatType referenceSet);

  /**
   * Set the reference tree to a new reference tree.
   */
  void Train(Tree* referenceTree);

  /**
   * Compute the rank approximate nearest neighbors of each query point in the
   * query set and store the output in the given matrices. The matrices will be
   * set to the size of n columns by k rows, where n is the number of points in
   * the query dataset and k is the number of neighbors being searched for.
   *
   * If querySet is small or only contains one point, it can be faster to do
   * single-tree search; single-tree search can be set with the SingleMode()
   * function or in the constructor.
   *
   * @param querySet Set of query points (can be a single point).
   * @param k Number of neighbors to search for.
   * @param neighbors Matrix storing lists of neighbors for each query point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void Search(const MatType& querySet,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * Compute the rank approximate nearest neighbors of each point in the
   * pre-built query tree and store the output in the given matrices. The
   * matrices will be set to the size of n columns by k rows, where n is the
   * number of points in the query dataset and k is the number of neighbors
   * being searched for.
   *
   * If singleMode or naive is enabled, then this method will throw a
   * std::invalid_argument exception; calling this function implies a dual-tree
   * algorithm.
   *
   * @note
   * If the tree type you are using modifies the data matrix, be aware that the
   * results returned from this function will be with respect to the modified
   * data matrix.
   *
   * @param queryTree Tree built on query points.
   * @param k Number of neighbors to search for.
   * @param neighbors Matrix storing lists of neighbors for each query point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void Search(Tree* queryTree,
              const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * Compute the rank approximate nearest neighbors of each point in the
   * reference set (that is, the query set is taken to be the reference set),
   * and store the output in the given matrices.  The matrices will be set to
   * the size of n columns by k rows, where n is the number of points in the
   * query dataset and k is the number of neighbors being searched for.
   *
   * @param k Number of neighbors to search for.
   * @param neighbors Matrix storing lists of neighbors for each point.
   * @param distances Matrix storing distances of neighbors for each query
   *      point.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& neighbors,
              arma::mat& distances);

  /**
   * This function recursively resets the RAQueryStat of the given query tree to
   * set 'bound' to SortPolicy::WorstDistance and 'numSamplesMade' to 0. This
   * allows a user to perform multiple searches with the same query tree,
   * possibly with different levels of approximation without requiring to build
   * a new pair of trees for every new (approximate) search.
   *
   * If Search() is called multiple times with the same query tree without
   * calling ResetQueryTree(), the results may not satisfy the theoretical
   * guarantees provided by the rank-approximate neighbor search algorithm.
   *
   * @param queryTree Tree whose statistics should be reset.
   */
  void ResetQueryTree(Tree* queryTree) const;

  //! Access the reference set.
  const MatType& ReferenceSet() const { return *referenceSet; }

  //! Get whether or not naive (brute-force) search is used.
  bool Naive() const { return naive; }
  //! Modify whether or not naive (brute-force) search is used.
  bool& Naive() { return naive; }

  //! Get whether or not single-tree search is used.
  bool SingleMode() const { return singleMode; }
  //! Modify whether or not single-tree search is used.
  bool& SingleMode() { return singleMode; }

  //! Get the rank-approximation in percentile of the data.
  double Tau() const { return tau; }
  //! Modify the rank-approximation in percentile of the data.
  double& Tau() { return tau; }

  //! Get the desired success probability.
  double Alpha() const { return alpha; }
  //! Modify the desired success probability.
  double& Alpha() { return alpha; }

  //! Get whether or not sampling is done at the leaves.
  bool SampleAtLeaves() const { return sampleAtLeaves; }
  //! Modify whether or not sampling is done at the leaves.
  bool& SampleAtLeaves() { return sampleAtLeaves; }

  //! Get whether or not we traverse to the first leaf without approximation.
  bool FirstLeafExact() const { return firstLeafExact; }
  //! Modify whether or not we traverse to the first leaf without approximation.
  bool& FirstLeafExact() { return firstLeafExact; }

  //! Get the limit on the size of a node that can be approximated.
  size_t SingleSampleLimit() const { return singleSampleLimit; }
  //! Modify the limit on the size of a node that can be approximation.
  size_t& SingleSampleLimit() { return singleSampleLimit; }

  //! Serialize the object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Permutations of reference points during tree building.
  std::vector<size_t> oldFromNewReferences;
  //! Pointer to the root of the reference tree.
  Tree* referenceTree;
  //! Reference dataset.  In some situations we may own this dataset.
  const MatType* referenceSet;

  //! If true, this object created the trees and is responsible for them.
  bool treeOwner;
  //! If true, we are responsible for deleting the dataset.
  bool setOwner;

  //! Indicates if naive random sampling on the set is being used.
  bool naive;
  //! Indicates if single-tree search is being used (opposed to dual-tree).
  bool singleMode;

  //! The rank-approximation in percentile of the data (between 0 and 100).
  double tau;
  //! The desired success probability (between 0 and 1).
  double alpha;
  //! Whether or not sampling is done at the leaves.  Faster, but less accurate.
  bool sampleAtLeaves;
  //! If true, we will traverse to the first leaf without approximation.
  bool firstLeafExact;
  //! The limit on the number of points in the largest node that can be
  //! approximated by sampling.
  size_t singleSampleLimit;

  //! Instantiation of distance metric.
  DistanceType distance;

  //! For access to mappings when building models.
  friend class LeafSizeRAWrapper<TreeType>;
}; // class RASearch

} // namespace mlpack

// Include implementation.
#include "ra_search_impl.hpp"

// Include convenient typedefs.
#include "ra_typedef.hpp"

#endif
