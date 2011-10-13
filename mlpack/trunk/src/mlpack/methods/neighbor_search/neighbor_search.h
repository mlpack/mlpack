/**
 * @file neighbor_search.h
 *
 * Defines the AllkNN class to perform all-k-nearest-neighbors on two specified
 * data sets.
 */
#ifndef __MLPACK_NEIGHBOR_SEARCH_H
#define __MLPACK_NEIGHBOR_SEARCH_H

#include <mlpack/core.h>
#include <mlpack/core/tree/bounds.h>
#include <mlpack/core/tree/spacetree.h>
#include <vector>
#include <string>

#include <mlpack/core/kernels/lmetric.h>
#include "sort_policies/nearest_neighbor_sort.h"

// Define CLI parameters for the NeighborSearch class.
PARAM_MODULE("neighbor_search",
    "Parameters for the distance-based neighbor search.");
PARAM_INT("k", "Number of neighbors to search for.", "neighbor_search", 5);
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
 * used and the sort function used.  Prototypes for the two policy classes are
 * given below.
 *
 * class Kernel {
 *  public:
 *   Kernel(); // Default constructor is necessary.
 *
 *   // Evaluate the kernel function on these two vectors.
 *   double Evaluate(arma::vec&, arma::vec&);
 * };
 *
 * class SortPolicy {
 *  public:
 *   // Return the index in the list where the new distance should be inserted,
 *   // or size_t() - 1 if it should not be inserted (i.e. if it is not any
 *   // better than any of the existing points in the list).  The list is sorted
 *   // such that the best point is first in the list.  The actual insertion is
 *   // not performed.
 *   static size_t SortDistance(arma::vec& list, double new_distance);
 *
 *   // Return whether or not value is "better" than ref.
 *   static inline bool IsBetter(const double value, const double ref);
 *
 *   // Return the best possible distance between two nodes.
 *   template<typename TreeType>
 *   static double BestNodeToNodeDistacne(TreeType* query_node,
 *                                        TreeType* reference_node);
 *
 *   // Return the best possible distance between a node and a point.
 *   template<typename TreeType>
 *   static double BestPointToNodeDistance(const arma::vec& query_point,
 *                                         TreeType* reference_node);
 *
 *   // Return the worst or best possible distance for this sort policy.
 *   static inline const double WorstDistance();
 *   static inline const double BestDistance();
 * };
 *
 * @tparam Kernel The kernel function.  Must provide a default constructor and
 *   a function 'double Evaluate(arma::vec&, arma::vec&)'.
 * @tparam SortPolicy The sort policy for distances.  See
 *   sort_policies/nearest_neighbor_sort.h for an example.
 */
template<typename Kernel = mlpack::kernel::SquaredEuclideanDistance,
         typename SortPolicy = NearestNeighborSort>
class NeighborSearch {

  /**
  * Extra data for each node in the tree.  For all nearest neighbors,
  * each node only
  * needs its upper bound on its nearest neighbor distances.
  */
  class QueryStat {

   public:
    /**
     * The bound on the node's neighbor distances.
     */
    double bound_;

    // In addition to any member variables for the statistic, all stat
    // classes need two Init
    // functions, one for leaves and one for non-leaves.

    /**
     * Initialization function used in tree-building when initializing
     * a leaf node.  For allnn, needs no additional information
     * at the time of tree building.
     */
    QueryStat() {
      // The bound starts at the worst possible distance.
      bound_ = SortPolicy::WorstDistance();
    }

  }; //class QueryStat

  // TreeType are BinarySpaceTrees where the data are bounded by
  // Euclidean bounding boxes, the data are stored in a Matrix,
  // and each node has a QueryStat for its bound.
  typedef tree::BinarySpaceTree<bound::HRectBound<2>, QueryStat> TreeType;

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
  std::vector<size_t> old_from_new_queries_;
  std::vector<size_t> old_from_new_references_;

  bool naive_;
  bool dual_mode_;

  // The number of points in a leaf
  size_t leaf_size_;

  // number of nearest neighbrs
  size_t knns_;

  // The total number of prunes.
  size_t number_of_prunes_;

  // The distance to the candidate nearest neighbor for each query
  arma::mat neighbor_distances_;

  // The indices of the candidate nearest neighbor for each query
  arma::Mat<size_t> neighbor_indices_;

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
  void ComputeNeighbors(arma::Mat<size_t>& resulting_neighbors,
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
  void ComputeSingleNeighborsRecursion_(size_t point_id, arma::vec& point,
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
  void InsertNeighbor(size_t query_index, size_t pos, size_t neighbor,
                      double distance);

}; // class NeighborSearch

}; // namespace neighbor
}; // namespace mlpack

// Include implementation.
#include "neighbor_search_impl.h"

// Include convenience typedefs.
#include "typedef.h"

#endif
