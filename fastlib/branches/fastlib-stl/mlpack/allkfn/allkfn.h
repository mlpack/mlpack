/**
 * @file allkfn.h
 *
 * Defines AllNN class to perform all-nearest-neighbors on two specified 
 * data sets.
 */

#ifndef ALLKFN_H
#define ALLKFN_H

#include <fastlib/fastlib.h>
#include <vector>
#include <functional>

#include <armadillo>

namespace mlpack {
namespace allkfn {

/**
 * Forward declaration for the tester class
 */
class TestAllkFN;

/**
* Performs all-furthest-neighbors.  This class will build the trees and 
* perform the recursive computation.
*/
class AllkFN {
  // Declare the tester class as a friend class so that it has access
  // to the private members of the class
  friend class TestAllkNFN;
  
  //////////////////////////// Nested Classes ///////////////////////////////////////////////
  /**
  * Extra data for each node in the tree.  For all nearest neighbors, 
  * each node only
  * needs its upper bound on its nearest neighbor distances.  
  */
  class QueryStat {
    
   private:
    
    /**
     * The lower bound on the node's nearest furthest distances.
     */
    double min_distance_so_far_;
    
   public:
    double min_distance_so_far() {
      return min_distance_so_far_; 
    } 
    
    void set_min_distance_so_far(double new_dist) {
      min_distance_so_far_ = new_dist; 
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
      // The bound starts at zero
      min_distance_so_far_ = 0;
    } 
    
    /**
     * Initialization function used in tree-building when initializing a non-leaf node.  For other algorithms,
     * node statistics can be built using information from the children.  
     */
    void Init(const arma::mat& matrix, index_t start, index_t count, 
        const QueryStat& left, const QueryStat& right) {
      // For allnn, non-leaves can be initialized in the same way as leaves
      Init(matrix, start, count);
    } 
    
  }; // class AllkFNStat  
  
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

  // The total number of prunes.
  index_t number_of_prunes_;

  // The number of points in a leaf
  index_t leaf_size_;

  // The distance to the candidate nearest neighbor for each query
  arma::vec neighbor_distances_;

  // The indices of the candidate nearest neighbor for each query
  arma::Col<index_t> neighbor_indices_;

  // number of furthest neighbrs
  index_t kfns_; 

 public:
  enum {
    NAIVE = 1,
    ALIAS_MATRIX = 2,
    MODE_SINGLE = 4
  };

  /**
   * Initialize the AllkFN object.  If only the reference matrix is given, the
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
   *      recommended and is faster in the vast majority of situations.
   *
   * @param queries_in Input matrix of query points
   * @param references_in Input matrix of reference points to query against
   * @param module_in Datanode containing input parameters
   * @param options Combination of options (NAIVE, ALIAS_MATRIX, MODE_SINGLE)
   * @param leaf_size Leaf size used for tree building (ignored in naive mode)
   * @param kfns Number of furthest neighbors to calculate
   */
  AllkFN(arma::mat& queries_in, arma::mat& references_in,
         struct datanode* module_in, int options = 0);
  AllkFN(arma::mat& references_in, struct datanode* module_in, int options = 0);
  AllkFN(arma::mat& queries_in, arma::mat& references_in, index_t leaf_size,
         index_t kfns, int options = 0);
  AllkFN(arma::mat& references_in, index_t leaf_size, index_t kfns,
         int options = 0);
  
  /**
  * The tree is the only member we are responsible for deleting.  The others
  * will take care of themselves.  
  */
  ~AllkFN();
    
      
 /////////////////////////////// Helper Functions ///////////////////////////////////////////////////
  
  /**
   * Computes the maximum squared distance between the bounding boxes of two nodes
   */
  double MaxNodeDistSq_(TreeType* query_node, TreeType* reference_node);
  
  /**
   * Performs exhaustive computation between two leaves.  
   */
  void ComputeBaseCase_(TreeType* query_node, TreeType* reference_node);
  
  /**
   * The recursive function for dual-mode computation.
   */
  void ComputeDualNeighborsRecursion_(TreeType* query_node,
                                      TreeType* reference_node, 
                                      double higher_bound_distance);
   
  /**
   * The recursive function for single-tree-mode computation.
   */
  void ComputeSingleNeighborsRecursion_(index_t pointId, arma::vec& point,
                                        TreeType* reference_node,
                                        double* max_dist_so_far);
  
  /**
   * Computes the nearest neighbors and stores them in *results
   */
  void ComputeNeighbors(arma::Col<index_t>& resulting_neighbors,
                        arma::vec& distances);
    
}; // class AllkFN

}; // namespace allkfn
}; // namespace mlpack

#endif
