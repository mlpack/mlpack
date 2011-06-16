/**
 * @file neighbor_search.h
 *
 * Defines the AllkNN class to perform all-k-nearest-neighbors on two specified
 * data sets.
 */

#ifndef __MLPACK_NEIGHBOR_SEARCH_H
#define __MLPACK_NEIGHBOR_SEARCH_H

#include <fastlib/fastlib.h>
#include <vector>
#include <string>

#include <mlpack/core/kernels/l2_squared_metric.h>
#include "sort_policies/nearest_neighbor_sort.h"

// Define IO parameters for the NeighborSearch class.
PARAM_MODULE("neighbor_search",
    "Parameters for the distance-based neighbor search.");
PARAM_INT("k", "Number of neighbors to search for.", "neighbor_search", 5);
PARAM_INT("leaf_size", "Leaf size for tree-building.", "neighbor_search", 20);
PARAM_FLAG("single_mode", "If set, use single-tree mode (instead of "
    "dual-tree).", "neighbor_search");
PARAM_FLAG("naive_mode", "If set, use naive computations (no trees).  This "
    "overrides the single_mode flag.", "neighbor_search");

namespace mlpack {
namespace neighbor {

/***
 * The NeighborSearch class is a template class for performing distance-based
 * neighbor searches.  It takes a query dataset and a reference dataset (or just
 * a reference dataset) and finds the k neighbors which have the 'best' distance
 * according to a given sorting policy.
 *
 * The template parameters Kernel and SortPolicy define the distance function
 * used and the sort function used.
 *
 * @tparam Kernel The kernel function.  Must provide a default constructor and
 *   a function 'void Evaluate(arma::vec&, arma::vec&)'.
 * @tparam SortPolicy The sort function for distances.  Must provide a function
 *   'index_t SortDistance(arma::vec&, double)'.  In this function a vector of
 *   distances is given (as well as a new distance) and the function must return
 *   the index in the vector where this distance should be inserted, or SIZE_MAX
 *   if it should not be inserted.
 */
template<typename Kernel = mlpack::kernel::L2SquaredMetric,
         typename SortPolicy = NearestNeighborSort>
class NeighborSearch {

  //////////////////////////// Nested Classes /////////////////////////////////
  /**
  * Extra data for each node in the tree.  For all nearest neighbors,
  * each node only
  * needs its upper bound on its nearest neighbor distances.
  */
  class QueryStat {

   private:
    /**
     * The upper bound on the node's nearest neighbor distances.
     */
    double max_distance_so_far_;

   public:
    double max_distance_so_far() {
      return max_distance_so_far_;
    }

    void set_max_distance_so_far(double new_dist) {
      max_distance_so_far_ = new_dist;
    }

    // In addition to any member variables for the statistic, all stat
    // classes need two Init
    // functions, one for leaves and one for non-leaves.

    /**
     * Initialization function used in tree-building when initializing
     * a leaf node.  For allnn, needs no additional information
     * at the time of tree building.
     */
    void Init(const arma::mat& matrix, index_t start, index_t count) {
      // The bound starts at the worst possible distance.
      max_distance_so_far_ = SortPolicy::WorstDistance();
    }

    /**
     * Initialization function used in tree-building when initializing a
     * non-leaf node.  For other algorithms, node statistics can be built using
     * information from the children.
     */
    void Init(const arma::mat& matrix, index_t start, index_t count,
        const QueryStat& left, const QueryStat& right) {
      // For allknn, non-leaves can be initialized in the same way as leaves
      Init(matrix, start, count);
    }

  }; //class AllNNStat

  // TreeType are BinarySpaceTrees where the data are bounded by
  // Euclidean bounding boxes, the data are stored in a Matrix,
  // and each node has a QueryStat for its bound.
  typedef BinarySpaceTree<DHrectBound<2>, arma::mat, QueryStat> TreeType;


  /////////////////////////////// Members /////////////////////////////////////
 private:
  // These will store our data sets.
  arma::mat references_;
  arma::mat queries_;

  // Instantiation of kernel (potentially not necessary).
  Kernel kernel_;

  // Pointers to the roots of the two trees.
  TreeType* reference_tree_;
  TreeType* query_tree_;

  // A permutation of the indices for tree building.
  arma::Col<index_t> old_from_new_queries_;
  arma::Col<index_t> old_from_new_references_;

  bool naive_;
  bool dual_mode_;

  // The number of points in a leaf
  index_t leaf_size_;

  // number of nearest neighbrs
  index_t knns_;

  // The total number of prunes.
  index_t number_of_prunes_;

  // The distance to the candidate nearest neighbor for each query
  arma::mat neighbor_distances_;

  // The indices of the candidate nearest neighbor for each query
  arma::Mat<index_t> neighbor_indices_;

 public:
  /**
   * Initialize the NeighborSearch object.  If only a reference dataset is
   * given, it is assumed that the query set is also the reference set.
   * Optionally, an initialized kernel can be passed in, for cases where the
   * kernel has internal data (i.e. the Mahalanobis distance).
   *
   * @param queries_in Set of query points.
   * @param references_in Set of reference points.
   * @param alias_matrix If true, alias the passed matrices instead of copying
   *     them.  While this lowers memory footprint and computational load, the
   *     matrices will be modified during the tree-building process!
   * @param kernel An optional instance of the Kernel class.
   */
  NeighborSearch(arma::mat& queries_in, arma::mat& references_in,
                 bool alias_matrix = false, Kernel kernel = Kernel());
  NeighborSearch(arma::mat& references_in, bool alias_matrix = false,
                 Kernel kernel = Kernel());

  /**
   * The tree is the only member we are responsible for deleting.  The others
   * will take care of themselves.
   */
  ~NeighborSearch();
  
  /**
   * Computes the nearest neighbors and stores the output in the given arrays.
   * For an AllkNN object with knns_ set to 5 (i.e. calculate the five nearest
   * neighbors), resulting_neighbors[0] through resulting_neighbors[4] are the
   * five nearest neighbors of query point 0.
   *
   * @param resulting_neighbors List of nearest neighbors
   * @param distances Distance of nearest neighbors
   */
  void ComputeNeighbors(arma::Mat<index_t>& resulting_neighbors,
                        arma::mat& distances);

 private:
  /**
   * Performs exhaustive computation between two leaves, comparing every node in
   * the leaf to the other leaf to find the furthest neighbor.  The
   * neighbor_indices_ and neighbor_distances_ arrays will be updated with the
   * changed information.
   */
  void ComputeBaseCase_(TreeType* query_node, TreeType* reference_node);

  /**
   * Recurse down the trees, computing base case computations when the leaves
   * are reached.
   */
  void ComputeDualNeighborsRecursion_(TreeType* query_node,
                                      TreeType* reference_node,
                                      double lower_bound);

  /***
   * Perform a recursion only on the reference tree; the query point is given.
   * This method is similar to ComputeBaseCase_().
   */
  void ComputeSingleNeighborsRecursion_(index_t point_id, arma::vec& point,
                                        TreeType* reference_node,
                                        double& best_dist_so_far);

  /***
   * Helper function to insert a point into the neighbors and distances
   * matrices.
   *
   * @param query_index Index of point whose neighbors we are inserting into.
   * @param pos Position in list to insert into.
   * @param neighbor Index of reference point which is being inserted.
   * @param distance Distance from query point to reference point.
   */
  void InsertNeighbor(index_t query_index, index_t pos, index_t neighbor,
                      double distance);

}; // class NeighborSearch

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "neighbor_search_impl.h"

// Include convenience typedefs.
#include "typedef.h"

#endif
