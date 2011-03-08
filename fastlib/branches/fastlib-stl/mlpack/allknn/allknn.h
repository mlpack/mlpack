/**
 * @file allknn.h
 *
 * Defines AllNN class to perform all-nearest-neighbors on two specified 
 * data sets.
 */

// inclusion guards, please add them to your .h files
#ifndef ALLKNN_H
#define ALLKNN_H

// We need to include fastlib.  If you want to use fastlib, 
// you need to have this line in addition to
// the deplibs section of your build.py
#include <fastlib/fastlib.h>
#include <vector>
#include <string>

namespace mlpack {
namespace allknn {

/**
 * Forward declaration for the tester class
 */
class TestAllkNN;
/**
* Performs all-nearest-neighbors.  This class will build the trees and 
* perform the recursive  computation.
*/
class AllkNN {
  // Declare the tester class as a friend class so that it has access
  // to the private members of the class
  friend class TestAllkNN;
  
  //////////////////////////// Nested Classes ///////////////////////////////////////////////
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
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
    } 
    
    /**
     * Initialization function used in tree-building when initializing a non-leaf node.  For other algorithms,
     * node statistics can be built using information from the children.  
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

  
  /////////////////////////////// Members //////////////////////////////////////////////////
 private:
  // The module containing the parameters for this computation. 
  struct datanode* module_;

  // These will store our data sets.
  arma::mat references_;
  arma::mat queries_;

  // Pointers to the roots of the two trees.
  TreeType* reference_tree_;
  TreeType* query_tree_;

  // A permutation of the indices for tree building.
  arma::Col<index_t> old_from_new_queries_;
  arma::Col<index_t> old_from_new_references_;

  bool naive_;
  bool dual_mode_;
 
 public: 
  // The total number of prunes.
  index_t number_of_prunes_;

 private:
  // The number of points in a leaf
  index_t leaf_size_;

  // The distance to the candidate nearest neighbor for each query
  arma::vec neighbor_distances_;

  // The indices of the candidate nearest neighbor for each query
  arma::Col<index_t> neighbor_indices_;

  // number of nearest neighbrs
  index_t knns_; 

  // if this flag is true then only the k-neighbor and distance are computed
  bool k_only_;

 public:
  enum {
    NAIVE = 1,
    ALIAS_MATRIX = 2,
    MODE_SINGLE = 4
  };

  /**
   * Initialize the AllkNN object.  If only the references matrix is given, the
   * queries matrix is assumed to be the same.
   *
   * The options parameter is meant to be a combination of options, such as
   * (NAIVE | ALIAS_MATRIX) or similar.  The three allowed options are:
   *
   *  - NAIVE: if set, the naive method for computation will be used
   *  - ALIAS_MATRIX: if set, the input matrices will be aliased internally.
   *      This will result in the input matrices being re-ordered while the
   *      trees are built.  You will get a performance boost from using this
   *      option, but it must be understood that the matrix you pass in will be
   *      modified.
   *  - MODE_SINGLE: if set, the single-tree method is used; otherwise, the
   *      dual-tree method is used (which is the default).  Dual-tree is
   *      recommended.
   *
   * @param queries_in Input matrix of query points
   * @param references_in Input matrix of reference points to query against
   * @param module_in Datanode containing input parameters
   * @param options Combination of options (NAIVE, ALIAS_MATRIX, MODE_SINGLE)
   * @param leaf_size Leaf size used for tree building (ignored in naive mode)
   * @param knns Number of nearest neighbors to calculate
   */
  AllkNN(arma::mat& queries_in, arma::mat& references_in,
         struct datanode* module_in, int options = 0);
  AllkNN(arma::mat& references_in, struct datanode* module_in, int options = 0);
  AllkNN(arma::mat& queries_in, arma::mat& references_in, index_t leaf_size,
         index_t knns, int options = 0);
  AllkNN(arma::mat& references_in, index_t leaf_size, index_t knns,
         int options = 0);
  
  /**
   * The tree is the only member we are responsible for deleting.  The others
   * will take care of themselves.  
   */
  ~AllkNN();  
      
  /**
   * Computes the minimum squared distance between the bounding boxes of two
   * nodes
   */
  double MinNodeDistSq_(TreeType* query_node, TreeType* reference_node);

  /**
   * Computes the minimum squared distances between a point and a node's
   * bounding box
   */
  double MinPointNodeDistSq_(const arma::vec& query_point,
                             TreeType* reference_node);
  
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
                                      double lower_bound_distance);
  
  /***
   * Perform a recursion only on the reference tree; the query point is given.
   * This method is similar to ComputeBaseCase_().
   */
  void ComputeSingleNeighborsRecursion_(index_t point_id, arma::vec& point,
                                        TreeType* reference_node,
                                        double* min_dist_so_far);
  
  /**
   * Computes the nearest neighbors and stores the output in the given arrays.
   * For an AllkNN object with knns_ set to 5 (i.e. calculate the five nearest
   * neighbors), resulting_neighbors[0] through resulting_neighbors[4] are the
   * five nearest neighbors of query point 0.
   *
   * @param resulting_neighbors List of nearest neighbors
   * @param distances Distance of nearest neighbors
   */
  void ComputeNeighbors(arma::Col<index_t>& resulting_neighbors,
                        arma::vec& distances);

}; // class AllkNN

}; // namespace allknn
}; // namespace mlpack

#endif
