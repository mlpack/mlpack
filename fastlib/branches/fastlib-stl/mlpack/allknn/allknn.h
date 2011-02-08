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
      // For allnn, non-leaves can be initialized in the same way as leaves
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
  arma::mat* queries_;
  arma::mat* references_;

  // Pointers to the roots of the two trees.
  TreeType* query_tree_;
  TreeType* reference_tree_;

  // The total number of prunes.
  index_t number_of_prunes_;

  // A permutation of the indices for tree building.
  arma::Col<index_t> old_from_new_queries_;
  arma::Col<index_t> old_from_new_references_;

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

  // This can be either "single" or "dual" referring to dual tree and single tree algorithm
  std::string mode_;
  
 public:

  /**
  * Constructors are generally very simple in FASTlib; most of the work is done by Init().  This is only
  * responsible for ensuring that the object is ready to be destroyed safely.  
  */
  AllkNN();
  
  /**
  * The tree is the only member we are responsible for deleting.  The others will take care of themselves.  
  */
  ~AllkNN();
    
      
  /**
   * Computes the minimum squared distance between the bounding boxes of two nodes
   */
  double MinNodeDistSq_ (TreeType* query_node, TreeType* reference_node);

  /**
   * Computes the minimum squared distances between a point and a node's bounding box
   */
  double MinPointNodeDistSq_ (const arma::vec& query_point, TreeType* reference_node);
  
  /**
   * Performs exhaustive computation between two leaves.  
   */
  void ComputeBaseCase_(TreeType* query_node, TreeType* reference_node);
  
  /**
   * The recursive function for dual tree
   */
  void ComputeDualNeighborsRecursion_(TreeType* query_node, TreeType* reference_node, 
      double lower_bound_distance);
  
  void ComputeSingleNeighborsRecursion_(index_t point_id, 
      arma::vec& point, TreeType* reference_node, 
      double* min_dist_so_far);
  
  /**
  * Setup the class and build the trees.  Note: we are initializing with const references to prevent 
  * local copies of the data.
  */
  void Init(arma::mat* queries_in, arma::mat* references_in, struct datanode* module_in);
  void Init(arma::mat* references_in, struct datanode* module_in);
  void Init(arma::mat* queries_in, arma::mat* references_in, index_t leaf_size, index_t knns, const char *mode="dual");
  void Init(arma::mat* references_in, index_t leaf_size, index_t knns, const char *mode="dual");
  
  /**
   * Initializes the AllNN structure for naive computation.  
   * This means that we simply ignore the tree building.
   */
  void Destruct();
  
  void InitNaive(arma::mat* queries_in, arma::mat* references_in, index_t knns);
  void InitNaive(arma::mat* references_in, index_t knns);
  
  /**
   * Computes the nearest neighbors and stores them in *results
   */
  void ComputeNeighbors(arma::Col<index_t>& resulting_neighbors,
                        arma::vec& distances);
  
  /**
   * Does the entire computation naively
   */
  void ComputeNaive(arma::Col<index_t>& resulting_neighbors,
                    arma::vec& distances);
}; // class AllNN

}; // namespace allknn
}; // namespace mlpack

#endif
