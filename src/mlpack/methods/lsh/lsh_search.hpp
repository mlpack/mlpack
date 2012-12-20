/**
 * @file lsh_search.hpp
 * @author Parikshit Ram
 *
 * Defines the LSHSearch class, which performs an abstract
 * distance-approximate nearest neighbor query on two datasets
 * using Locality-sensitive hashing with 2-stable distributions
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_SEARCH_HPP

#include <mlpack/core.hpp>
#include <vector>
#include <string>

#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>

namespace mlpack {
namespace neighbor /** Neighbor-search routines.  These include
                    * all-nearest-neighbors and all-furthest-neighbors
                    * searches. */ {

/**
 * The RASearch class is a template class for performing distance-based
 * neighbor searches.  It takes a query dataset and a reference dataset (or just
 * a reference dataset) and, for each point in the query dataset, finds the k
 * neighbors in the reference dataset which have the 'best' distance according
 * to a given sorting policy.  A constructor is given which takes only a
 * reference dataset, and if that constructor is used, the given reference
 * dataset is also used as the query dataset.
 *
 * The template parameters SortPolicy and Metric define the sort function used
 * and the metric (distance function) used.  More information on those classes
 * can be found in the NearestNeighborSort class and the kernel::ExampleKernel
 * class.
 *
 * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
 * @tparam MetricType The metric to use for computation.
 * @tparam TreeType The tree type to use.
 */
template<typename SortPolicy = NearestNeighborSort,
         typename MetricType = mlpack::metric::SquaredEuclideanDistance,
         typename eT = double>

class LSHSearch
{
 public:

  typedef arma::Mat<eT> MatType;
  typedef arma::Col<eT> ColType;
  typedef arma::Row<eT> RowType;

  /**
   * Initialize the RASearch object, passing both a query and reference
   * dataset.  Optionally, perform the computation in naive mode or single-tree
   * mode, and set the leaf size used for tree-building.  An initialized
   * distance metric can be given, for cases where the metric has internal data
   * (i.e. the distance::MahalanobisDistance class).
   *
   * This method will copy the matrices to internal copies, which are rearranged
   * during tree-building.  You can avoid this extra copy by pre-constructing
   * the trees and passing them using a diferent constructor.
   *
   * @param referenceSet Set of reference points.
   * @param querySet Set of query points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param leafSize Leaf size for tree construction (ignored if tree is given).
   * @param metric An optional instance of the MetricType class.
   */
  LSHSearch(const MatType& referenceSet,
            const MatType& querySet,
            const size_t numProj,
            const size_t numTables,
            const double hashWidth,
            const size_t secondHashSize = 99901,
            const size_t bucketSize = 500,
            const MetricType metric = MetricType());

  /**
   * Initialize the RASearch object, passing only one dataset, which is
   * used as both the query and the reference dataset.  Optionally, perform the
   * computation in naive mode or single-tree mode, and set the leaf size used
   * for tree-building.  An initialized distance metric can be given, for cases
   * where the metric has internal data (i.e. the distance::MahalanobisDistance
   * class).
   *
   * If naive mode is being used and a pre-built tree is given, it may not work:
   * naive mode operates by building a one-node tree (the root node holds all
   * the points).  If that condition is not satisfied with the pre-built tree,
   * then naive mode will not work.
   *
   * @param referenceSet Set of reference points.
   * @param naive If true, O(n^2) naive search will be used (as opposed to
   *      dual-tree search).  This overrides singleMode (if it is set to true).
   * @param singleMode If true, single-tree search will be used (as opposed to
   *      dual-tree search).
   * @param leafSize Leaf size for tree construction (ignored if tree is given).
   * @param metric An optional instance of the MetricType class.
   */
  LSHSearch(const MatType& referenceSet,
            const size_t numProj,
            const size_t numTables,
            const double hashWidth,
            const size_t secondHashSize = 99901,
            const size_t bucketSize = 500,
            const MetricType metric = MetricType());
  /**
   * Delete the RASearch object. The tree is the only member we are
   * responsible for deleting.  The others will take care of themselves.
   */
  ~LSHSearch();

  /**
   * Compute the nearest neighbors and store the output in the given matrices.
   * The matrices will be set to the size of n columns by k rows, where n is the
   * number of points in the query dataset and k is the number of neighbors
   * being searched for.
   *
   * @param k Number of neighbors to search for.
   * @param resultingNeighbors Matrix storing lists of neighbors for each query
   *     point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void Search(const size_t k,
              arma::Mat<size_t>& resultingNeighbors,
              arma::mat& distances);

 private:

  void BuildFirstLevelHash(MatType* allKeyPointMat);

  void BuildSecondLevelHash(MatType& allKeyPointMat);

  inline void BaseCase(const size_t queryIndex, 
                       const size_t referenceIndex);

  void InsertNeighbor(const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double distance);

  void ReturnIndicesFromTable(const size_t queryIndex,
                              arma::uvec& referenceIndices);


 private:
  //! Reference dataset.
  const arma::mat& referenceSet;
  //! Query dataset (may not be given).
  const arma::mat& querySet;

  //! Instantiation of kernel.
  MetricType metric;

  //! The number of projections
  const size_t numProj;

  //! The number of tables
  const size_t numTables;

  //! The std::vector containing the projection matrix of each table
  std::vector<MatType> projections; // should be [numProj x dims] x numTables

  //! The list of the offset 'b' for each of the projection for each table
  MatType offsets; // should be numProj x numTables

  //! The hash width
  const double hashWidth;

  //! The big prime representing the size of the second hash
  const size_t secondHashSize;

  //! The weights of the second hash
  ColType secondHashWeights;

  //! The bucket size of the second hash
  const size_t bucketSize;

  //! The final hash table
  arma::Mat<size_t> secondHashTable; // should be (< secondHashSize) x bucketSize

  //! The number of elements present in each hash bucket
  arma::Col<size_t> bucketContentSize; // should be secondHashSize

  //! For a particular hash value, points to the row in secondHashTable
  //! corresponding to this value
  arma::Col<size_t> bucketRowInHashTable; // should be secondHashSize

  //! The pointer to the nearest neighbor distance
  arma::mat* distancePtr;

  //! The pointer to the nearest neighbor indices
  arma::Mat<size_t>* neighborPtr;


}; // class LSHSearch

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "lsh_search_impl.hpp"

// Include convenience typedefs.
//#include "lsh_typedef.hpp"

#endif
