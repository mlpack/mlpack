/**
 * @file allnn.h
 *
 * This file contains the definition of the AllNN class for dual-tree or naive
 * all-nearest-neighbors computation.
 * 
 * @see allnn_main.cc
 */

#ifndef MLPACK_ALLNN_H_
#define MLPACK_ALLNN_H_
#include "fastlib/fastlib.h"

namespace mlpack {
namespace allnn {

/**
 * A computation class for dual-tree and naive all-nearest-neighbors.
 *
 * This class builds trees for (assumed distinct) input query and
 * reference sets on Init.  It can also handle monochromatic cases
 * where the query and the reference trees are the same
 * The all-nearest-neighbors computation is
 * then performed by calling ComputeNeighbors or ComputeNaive.
 *
 * This class is only intended to compute once per instantiation.
 *
 * Example use:
 *
 * @code
 *   AllNN allnn;
 *   struct datanode* allnn_module;
 *   arma::Col<index_t> results;
 *
 *   allnn_module = fx_submodule(NULL, "allnn", "allnn");
 *   allnn.Init(query_set, reference_set, allnn_module);
 *   allnn.ComputeNeighbors(results);
 * @endcode
 */
class AllNN {

 ////////// Nested Classes //////////////////////////////////////////
 private:
  /**
   * Additional data stored at each node of a BinarySpaceTree, used by
   * our QueryTree type.
   *
   * Dual-tree all-nearest-neighbors maintains an upper bound on
   * nearest neighbor distance for each query node.
   */
  class QueryStat {

   private:
    /**
     * An upper bound on nearest neighbor distance for points within
     * the node, to be modified after tree formation.
     */
    double max_distance_so_far_;
    
   public:
    QueryStat() { }   
    
    double max_distance_so_far() { return max_distance_so_far_; }
    void set_max_distance_so_far(double new_dist) {
      max_distance_so_far_ = new_dist;
    }
    
    /**
     * An Init function required by BinarySpaceTree to build
     * statistics for a leaf node.
     *
     * All-nearest-neighbors fills statistics during computation, but
     * we must set it to an appropriate starting value here.
     */
    void Init(const arma::mat& matrix, index_t start, index_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
   }

    /**
     * An Init function required by BinarySpaceTree to build
     * statistics from two child nodes.
     *
     * For all-nearest-neighbors, we reuse initialization for leaves.
     */
    void Init(const arma::mat& matrix, index_t start, index_t count,
	      const QueryStat& left, const QueryStat& right) {
      Init(matrix, start, count);
    }
  }; /* class QueryStat */

  // DHrectBound<2> gives us the normal kind of kd-tree bounding boxes, using
  // the 2-norm, and arma::mat specifies the storage type of our data.

  /** kd-tree (binary with hrect bounds) with stats for queries. */
  typedef BinarySpaceTree<DHrectBound<2>, arma::mat, QueryStat> TreeType;

  ////////// Members Variables ///////////////////////////////////////

 private:
  arma::mat references_;  /// Matrix of reference points.
  arma::mat queries_;     /// Matrix of query points.

  bool naive_; /// Whether or not naive computation is being used.

  /** Root of a tree formed on references_. */
  TreeType* reference_tree_;
  /** Root of a tree formed on queries_. */
  TreeType* query_tree_;


  /** Maximum number of points in either tree's leaves. */
  index_t leaf_size_;

  /** Permutation mapping indices of queries_ to original order. */
  arma::Col<index_t> old_from_new_queries_;
  /** Permutation mapping indices of references_ to original order. */
  arma::Col<index_t> old_from_new_references_;

  /**
   * Candidate nearest neighbor distances, modified during
   * computation.  Later, true nearest neighbor distances.
   */
  arma::vec neighbor_distances_;
  /**
   * Candidate nearest neighbor indicies, modified during
   * compuatation.  Later, true nearest neighbor indices.
   */
  arma::Col<index_t> neighbor_indices_;

  /** Number of node-pairs pruned by the dual-tree algorithm. */
  index_t number_of_prunes_;
  
  /** Debug-mode test whether an AllNN object is initialized. */
  bool initialized_;
  /** Debug-mode test whether an AllNN object is used twice. */
  bool already_used_;




 public:
  // Constructors

  /***
   * Read parameters, and build the trees.  We assume that the queries are the
   * same as the references if this constructor is used.
   * 
   * This method will, by default, copy the input matrix to an internal matrix
   * which will have its ordering shuffled due to the tree-building procedure.
   * You can force AllNN to alias the input matrix, resulting in a performance
   * gain, by setting alias_matrix = true.  <b>However</b>, the ordering of the
   * input matrix will be changed completely after this constructor is called.
   * You have been warned!
   * 
   * @param[in] references_in Input matrix; for this constructor we assume the
   *   queries are the same as the references
   * @param[in] module_in Datanode holding input parameters.
   * @param[in] alias_matrix If set to true, alias the matrix instead of copying
   *   it.
   * @param[in] naive Use naive (non-tree-based) nearest neighbors calculation.
   */
  AllNN(arma::mat& references_in, bool alias_matrix = false, bool naive = false);

  /***
   * Read parameters, and build the trees.
   * 
   * This method will, by default, copy the input matrix to an internal matrix
   * which will have its ordering shuffled due to the tree-building procedure.
   * You can force AllNN to alias the input matrix, resulting in a performance
   * gain, by setting alias_matrix = true.  <b>However</b>, the ordering of the
   * input matrix will be changed completely after this constructor is called.
   * You have been warned!
   *
   * @param[in] queries_in Input matrix of query points
   * @param[in] references_in Input matrix of reference points
   * @param[in] module_in Datanode holding input parameters
   * @param[in] alias_matrix If set to true, alias the matrix instead of copying
   *   it.
   * @param[in] naive Use naive (non-tree-based) nearest neighbors calculation.
   */
  AllNN(arma::mat& queries_in, arma::mat& references_in, 
		bool alias_matrix = false, bool naive = false);

  ~AllNN();

 private:
  // Helper functions

  /**
   * Computes the minimum squared distance between the bounding boxes
   * of two nodes.
   */
  double MinNodeDistSq_(TreeType* query_node, TreeType* reference_node);

  /**
   * Performs exhaustive computation between two nodes.
   *
   * Note that naive also makes use of this function.
   */
  void GNPBaseCase_(TreeType* query_node, TreeType* reference_node);

  /**
   * Performs one node-node comparison in the GNP algorithm and
   * recurses upon all child combinations if no prune.
   */
  void GNPRecursion_(TreeType* query_node, TreeType* reference_node,
		     double lower_bound_distance);

 public:
  // Public functions

  /**
   * Computes the nearest neighbors and stores them in results if
   * provided.  Overloaded version provided if you don't have any results you
   * are interested in.
   */
  void ComputeNeighbors(arma::vec& distances);
  void ComputeNeighbors(arma::vec& distances, arma::Col<index_t>& results);

  /**
   * Initialize and fill an arma::vec of results.
   */
  void EmitResults(arma::vec& distances, arma::Col<index_t>& results);
 
  static void loadDocumentation();

}; /* class AllNN */
}; // namespace allnn
}; // namespace mlpack

#endif 
