/**
 * @file exact_max_ip.h
 *
 */

#ifndef EXACT_MAX_IP_H
#define EXACT_MAX_IP_H

#include <assert.h>
#include <fastlib/fastlib.h>
#include <vector>
#include <armadillo>
#include "general_spacetree.h"
#include "gen_metric_tree.h"
#include "dconebound.h"
#include "gen_cosine_tree.h"

using namespace mlpack;

PARAM_MODULE("maxip", "Parameters for the class that "
	     "builds a tree on the reference set and "
	     "searches for the maximum inner product "
	     "by the branch-and-bound method.");

PARAM_INT("knns", "The number of top innner products required",
	  "maxip", 1);
PARAM_DOUBLE("tau", "The rank error in terms of the \% of "
	     "reference set size", "maxip", 1.0);
PARAM_DOUBLE("alpha", "The error probability", "maxip", 0.95);
PARAM_INT("leaf_size", "The leaf size for the ball-tree", 
	  "maxip", 20);

PARAM_FLAG("angle_prune", "The flag to trigger the tighter"
	   " pruning using the angles as well", "maxip");
PARAM_FLAG("dual_tree", "The flag to trigger dual-tree "
	   "computation, using a cosine tree for the "
	   "queries.", "maxip");

PARAM_FLAG("check_prune", "The flag to trigger the "
	   "checking of the prune.", "maxip");



//   {"tree_building", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to record the time taken to build" 
//    " the query and the reference tree.\n"},
//   {"tree_building_approx", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to record the time taken to build" 
//    " the query and the reference tree for InitApprox.\n"},
//   {"computing_sample_sizes", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to compute the sample sizes.\n"},


/**
 * Performs maximum-inner-product-search. 
 * This class will build the trees and 
 * perform the recursive  computation.
 */
class MaxIP {
  
  //////////////////////////// Nested Classes /////////////////////////
  class QueryStat {
  private:
    double bound_;

  public:
    double bound() { return bound_; }
    void set_bound(double bound) { 
      //if (bound_ < bound) 
      bound_ = bound;
    }

    QueryStat() {
      bound_ = 0.0;
    }

    ~QueryStat() {}

    void Init(const arma::mat& data, size_t begin, size_t count) {
      bound_ = 0.0;
    }

    void Init(const arma::mat& data, size_t begin, size_t count,
	      QueryStat& left_stat, QueryStat& right_stat) {
      bound_ = 0.0;
    }
  }; // QueryStat

  // TreeType are BinarySpaceTrees where the data are bounded by 
  // Euclidean bounding boxes, the data are stored in a Matrix, 
  // and each node has a QueryStat for its bound.
  typedef GeneralBinarySpaceTree<DBallBound<>, arma::mat> TreeType;
  typedef GeneralBinarySpaceTree<DConeBound<>, arma::mat, QueryStat> CTreeType;
   
  
  /////////////////////////////// Members ////////////////////////////
private:
  // These will store our data sets.
  arma::mat queries_;
  arma::mat references_;

  // This will store the query index for the single tree run
  size_t query_;
  arma::vec query_norms_;

  // Pointers to the roots of the two trees.
  TreeType* reference_tree_;
  CTreeType* query_tree_;

  // The total number of prunes.
  size_t number_of_prunes_;

  // A permutation of the indices for tree building.
  arma::Col<size_t> old_from_new_queries_;
  arma::Col<size_t> old_from_new_references_;

  // The number of points in a leaf
  size_t leaf_size_;
  // number of nearest neighbrs
  size_t knns_; 

  // The distance to the candidate nearest neighbor for each query
  arma::vec max_ips_;
  // The indices of the candidate nearest neighbor for each query
  arma::Col<size_t> max_ip_indices_;

  // The total number of distance computations
  size_t distance_computations_;
  // The total number of split decisions
  size_t split_decisions_;


  /////////////////////////////// Constructors ////////////////////////
  
  // Add this at the beginning of a class to prevent accidentally
  // calling the copy constructor
  // FORBID_ACCIDENTAL_COPIES(MaxIP);
  
public:
  /**
   * Constructors are generally very simple in FASTlib;
   * most of the work is done by Init().  This is only
   * responsible for ensuring that the object is ready
   * to be destroyed safely.  
   */
  MaxIP() {
    reference_tree_ = NULL;
    query_tree_ = NULL;
  } 
  
  /**
   * The tree is the only member we are responsible for deleting.
   * The others will take care of themselves.  
   */
  ~MaxIP() {
    if (reference_tree_ != NULL) 
      delete reference_tree_;
 
    if (query_tree_ != NULL)
      delete query_tree_;
  }
    
  /////////////////////////// Helper Functions //////////////////////
  
private:
  /**
   * Computes the maximum inner product possible 
   * between the query point and the reference node.
   */
  double MaxNodeIP_(TreeType* reference_node);

  /**
   * Dual-tree: Computes the maximum inner product possible 
   * between the query node and the reference node while 
   * ignoring the norm of any query.
   * So it is computing \max_(q,r) |r| cos <qr.
   */
  double MaxNodeIP_(CTreeType *query_node, TreeType* reference_node);

  /**
   * Performs exhaustive computation at the leaves.  
   */
  void ComputeBaseCase_(TreeType* reference_node);

  /**
   * Dual-tree: Performs exhaustive computation between two leaves.  
   */
  void ComputeBaseCase_(CTreeType* query_node, TreeType* reference_node);
  
  /**
   * The recursive function
   */
  void ComputeNeighborsRecursion_(TreeType* reference_node, 
				  double upper_bound_ip);

  /**
   * Dual-tree: The recursive function
   */
  void ComputeNeighborsRecursion_(CTreeType* query_node,
				  TreeType* reference_node, 
				  double upper_bound_ip);

  /////////////// Public Functions ////////////////////
public:
  /**
   * Setup the class and build the trees.
   * Note: we are initializing with const references to prevent 
   * local copies of the data.
   */
  void Init(const arma::mat& queries_in, const arma::mat& references_in);

  void Destruct() {
    if (reference_tree_ != NULL)
      delete reference_tree_;
    
    if (query_tree_ != NULL)
      delete query_tree_;
  }

  /**
   * Initializes the AllNN structure for naive computation.  
   * This means that we simply ignore the tree building.
   */
  void InitNaive(const arma::mat& queries_in, 
		 const arma::mat& references_in);
  
  /**
   * Computes the nearest neighbors and stores them in *results
   */
  double ComputeNeighbors(arma::Col<size_t>* resulting_neighbors,
                        arma::vec* ips);
  
  /**
   * Does the entire computation naively
   */
  double ComputeNaive(arma::Col<size_t>* resulting_neighbors,
		    arma::vec* ips);


  void CheckPrune(CTreeType* query_node, TreeType* ref_node);
}; //class MaxIP

#endif
