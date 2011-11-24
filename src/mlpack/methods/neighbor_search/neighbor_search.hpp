/**
 * @file neighbor_search.hpp
 * @author Ryan Curtin
 *
 * Defines the AllkNN class to perform all-k-nearest-neighbors on two specified
 * data sets.
 */
#ifndef __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP
#define __MLPACK_METHODS_NEIGHBOR_SEARCH_NEIGHBOR_SEARCH_HPP

#include <mlpack/core.h>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <vector>
#include <string>

#include <mlpack/core/metrics/lmetric.hpp>
#include "sort_policies/nearest_neighbor_sort.hpp"

namespace mlpack {
namespace neighbor /** Neighbor-search routines.  These include
                    * all-nearest-neighbors and all-furthest-neighbors
                    * searches. */ {

// Define CLI parameters for the NeighborSearch class.
PARAM_MODULE("neighbor_search",
    "Parameters for the distance-based neighbor search.");
PARAM_INT("k", "Number of neighbors to search for.", "neighbor_search", 5);
PARAM_FLAG("single_mode", "If set, use single-tree mode (instead of "
    "dual-tree).", "neighbor_search");
PARAM_FLAG("naive_mode", "If set, use naive computations (no trees).  This "
    "overrides the single_mode flag.", "neighbor_search");

/**
 * The NeighborSearch class is a template class for performing distance-based
 * neighbor searches.  It takes a query dataset and a reference dataset (or just
 * a reference dataset) and, for each point in the query dataset, finds the k
 * neighbors in the reference dataset which have the 'best' distance according
 * to a given sorting policy.  A constructor is given which takes only a
 * reference dataset, and if that constructor is used, the given reference
 * dataset is also used as the query dataset.
 *
 * The template parameters Kernel and SortPolicy define the distance function
 * used and the sort function used.  More information on those classes can be
 * found in the kernel::ExampleKernel class and the NearestNeighborSort class.
 *
 * This class has several parameters configurable by the CLI interface:
 *
 * @param neighbor_search/k Parameters for the distance-based neighbor search.
 * @param neighbor_search/single_mode If set, single-tree mode will be used
 *     (instead of dual-tree mode).
 * @param neighbor_search/naive_mode If set, naive computation will be used (no
 *     trees). This overrides the single_mode flag.
 *
 * @tparam Kernel The kernel function; see kernel::ExampleKernel.
 * @tparam SortPolicy The sort policy for distances; see NearestNeighborSort.
 */
template<typename Kernel = mlpack::metric::SquaredEuclideanDistance,
         typename SortPolicy = NearestNeighborSort>
class NeighborSearch
{
  /**
   * Extra data for each node in the tree.  For neighbor searches, each node
   * only needs to store a bound on neighbor distances.
   */
  class QueryStat
  {
   public:
    //! The bound on the node's neighbor distances.
    double bound_;

    /**
     * Initialize the statistic with the worst possible distance according to
     * our sorting policy.
     */
    QueryStat() : bound_(SortPolicy::WorstDistance()) { }
  };

  /**
   * Simple typedef for the trees, which use a bound and a QueryStat (to store
   * distances for each node).  The bound should be configurable...
   */
  typedef tree::BinarySpaceTree<bound::HRectBound<2>, QueryStat> TreeType;

 private:
  //! Reference dataset.
  arma::mat references_;
  //! Query dataset (may not be given).
  arma::mat queries_;

  //! Instantiation of kernel.
  Kernel kernel_;

  //! Pointer to the root of the reference tree.
  TreeType* reference_tree_;
  //! Pointer to the root of the query tree (might not exist).
  TreeType* query_tree_;

  //! Permutations of query points during tree building.
  std::vector<size_t> old_from_new_queries_;
  //! Permutations of reference points during tree building.
  std::vector<size_t> old_from_new_references_;

  //! Indicates if O(n^2) naive search is being used.
  bool naive_;
  //! Indicates if dual-tree search is being used (opposed to single-tree).
  bool dual_mode_;

  //! Number of points in a leaf.
  size_t leaf_size_;

  //! Number of neighbors to compute.
  size_t knns_;

  //! Total number of pruned nodes during the neighbor search.
  size_t number_of_prunes_;

  //! The distance to the candidate nearest neighbor for each query
  arma::mat neighbor_distances_;

  //! The indices of the candidate nearest neighbor for each query
  arma::Mat<size_t> neighbor_indices_;

 public:
  /**
   * Initialize the NeighborSearch object, passing both a query and reference
   * dataset.  An initialized distance metric can be given, for cases where the
   * metric has internal data (i.e. the distance::MahalanobisDistance class).
   *
   * @param queries_in Set of query points.
   * @param references_in Set of reference points.
   * @param alias_matrix If true, alias the passed matrices instead of copying
   *     them.  While this lowers memory footprint and computational load, the
   *     points in the matrices will be rearranged during the tree-building
   *     process!  Defaults to false.
   * @param kernel An optional instance of the Kernel class.
   */
  NeighborSearch(arma::mat& queries_in, arma::mat& references_in,
                 bool alias_matrix = false, Kernel kernel = Kernel());

  /**
   * Initialize the NeighborSearch object, passing only one dataset.  In this
   * case, the query dataset is equivalent to the reference dataset, with one
   * caveat: for any point, the returned list of neighbors will not include
   * itself.  An initialized distance metric can be given, for cases where the
   * metric has internal data (i.e. the distance::MahalanobisDistance class).
   *
   * @param references_in Set of reference points.
   * @param alias_matrix If true, alias the passed matrices instead of copying
   *     them.  While this lowers memory footprint and computational load, the
   *     points in the matrices will be rearranged during the tree-building
   *     process!  Defaults to false.
   * @param kernel An optional instance of the Kernel class.
   */
  NeighborSearch(arma::mat& references_in, bool alias_matrix = false,
                 Kernel kernel = Kernel());

  /**
   * Delete the NeighborSearch object. The tree is the only member we are
   * responsible for deleting.  The others will take care of themselves.
   */
  ~NeighborSearch();

  /**
   * Compute the nearest neighbors and store the output in the given matrices.
   * The matrices will be set to the size of n columns by k rows, where n is the
   * number of points in the query dataset and k is the number of neighbors
   * being searched for.
   *
   * The parameter k is set through the CLI interface, not in the arguments to
   * this method; this allows users to specify k on the command line
   * ("--neighbor_search/k").  See the CLI documentation for further information
   * on how to use this functionality.
   *
   * @param resulting_neighbors Matrix storing lists of neighbors for each query
   *     point.
   * @param distances Matrix storing distances of neighbors for each query
   *     point.
   */
  void ComputeNeighbors(arma::Mat<size_t>& resulting_neighbors,
                        arma::mat& distances);

 private:
  /**
   * Perform exhaustive computation between two leaves, comparing every node in
   * the leaf to the other leaf to find the furthest neighbor.  The
   * neighbor_indices_ and neighbor_distances_ matrices will be updated with the
   * changed information.
   *
   * @param query_node Node in query tree.  This should be a leaf
   *     (bottom-level).
   * @param reference_node Node in reference tree.  This should be a leaf
   *     (bottom-level).
   */
  void ComputeBaseCase_(TreeType* query_node, TreeType* reference_node);

  /**
   * Recurse down the trees, computing base case computations when the leaves
   * are reached.
   *
   * @param query_node Node in query tree.
   * @param reference_node Node in reference tree.
   * @param lower_bound The lower bound; if above this, we can prune.
   */
  void ComputeDualNeighborsRecursion_(TreeType* query_node,
                                      TreeType* reference_node,
                                      double lower_bound);

  /**
   * Perform a recursion only on the reference tree; the query point is given.
   * This method is similar to ComputeBaseCase_().
   *
   * @param point_id Index of query point.
   * @param point The query point.
   * @param reference_node Reference node.
   * @param best_dist_so_far Best distance to a node so far -- used for pruning.
   */
  void ComputeSingleNeighborsRecursion_(size_t point_id, arma::vec& point,
                                        TreeType* reference_node,
                                        double& best_dist_so_far);

  /**
   * Insert a point into the neighbors and distances matrices; this is a helper
   * function.
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
#include "neighbor_search_impl.hpp"

// Include convenience typedefs.
#include "typedef.hpp"

#endif
