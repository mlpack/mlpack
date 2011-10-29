/**
 * @file exact_max_ip.h
 *
 */

#ifndef EXACT_MAX_IP_H
#define EXACT_MAX_IP_H

//#define NDEBUG

#include <assert.h>
#include <mlpack/core.h>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/statistic.hpp>
#include <vector>
#include <armadillo>
#include "general_spacetree.h"
#include "gen_metric_tree.h"
#include "dconebound.h"
#include "gen_cone_tree.h"

using namespace mlpack;

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
    double center_norm_;

  public:
    double bound() { return bound_; }
    double center_norm() { return center_norm_; }

    void set_bound(double bound) { 
      bound_ = bound;
    }

    void set_center_norm(double val) {
      center_norm_ = val; 
    }

    QueryStat() {
      bound_ = 0.0;
      center_norm_ = 0.0;
    }

    ~QueryStat() {}

    void Init(const arma::mat& data, size_t begin, size_t count) {
      bound_ = 0.0;
      center_norm_ = 0.0;
    }

    void Init(const arma::mat& data, size_t begin, size_t count,
	      QueryStat& left_stat, QueryStat& right_stat) {
      bound_ = 0.0;
      center_norm_ = 0.0;
    }
  }; // QueryStat

  class RefStat {
  private:
    double cosine_origin_;
    double sine_origin_;
    double dist_to_origin_;

  public:
    double cosine_origin() { return cosine_origin_; }
    double sine_origin() { return sine_origin_; }
    double dist_to_origin() { return dist_to_origin_; }


    void set_angles(double val, size_t type = 0) { 
      if (type == 0) { // given value is the cosine
	cosine_origin_ = val;
	sine_origin_ = std::sqrt(1 - val * val);
      } else { // the given value is the sine
	sine_origin_ = val;
	cosine_origin_ = std::sqrt(1 - val * val);
      }
    }

    void set_dist_to_origin(double val) {
      dist_to_origin_ = val;
    }

    // FILL OUT THE INIT FUNCTIONS APPROPRIATELY LATER
    RefStat() {
      cosine_origin_ = 0.0;
      sine_origin_ = 0.0;
      dist_to_origin_ = 0.0;
    }

    ~RefStat() {}
    void Init(const arma::mat& data, size_t begin, size_t count) {
      cosine_origin_ = 0.0;
      sine_origin_ = 0.0;
      dist_to_origin_ = 0.0;
    }

    void Init(const arma::mat& data, size_t begin, size_t count,
	      RefStat& left_stat, RefStat& right_stat) {
      cosine_origin_ = 0.0;
      sine_origin_ = 0.0;
      dist_to_origin_ = 0.0;
    }
  }; // RefStat

  // TreeType are BinarySpaceTrees where the data are bounded by 
  // Euclidean bounding boxes, the data are stored in a Matrix, 
  // and each node has a QueryStat for its bound.
  typedef GeneralBinarySpaceTree<bound::DBallBound<>, arma::mat, RefStat> TreeType;
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
  size_t ball_has_origin_;
  size_t cone_has_centroid_;

  // A permutation of the indices for tree building.
  arma::Col<size_t> old_from_new_queries_;
  arma::Col<size_t> old_from_new_references_;

  // The number of points in a leaf
  size_t leaf_size_;
  // number of nearest neighbrs
  size_t knns_; 

  // The distance to the candidate nearest neighbor for each query
  arma::mat max_ips_;
  // The indices of the candidate nearest neighbor for each query
  arma::Mat<size_t> max_ip_indices_;

  // The total number of distance computations
  size_t distance_computations_;
  // The total number of split decisions
  size_t split_decisions_;


  /////////////////////////////// Constructors ////////////////////////
  
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

  void reset_tree_(CTreeType *tree);
  void set_angles_in_balls_(TreeType *tree);
  void set_norms_in_cones_(CTreeType *tree);

  size_t SortValue(double value);

  void InsertNeighbor(size_t pos, size_t point_ind, double value);

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

  /*
   *
   */
  void WarmInit(size_t knns);

  /**
   * Computes the nearest neighbors and stores them in *results
   */
  double ComputeNeighbors(arma::Mat<size_t>* resulting_neighbors,
			  arma::mat* ips);
  
  /**
   * Does the entire computation naively
   */
  double ComputeNaive(arma::Mat<size_t>* resulting_neighbors,
		      arma::mat* ips);


  void CheckPrune(CTreeType* query_node, TreeType* ref_node);
}; //class MaxIP


PARAM_MODULE("maxip", "Parameters for the class that "
	     "builds a tree on the reference set and "
	     "searches for the maximum inner product "
	     "by the branch-and-bound method.");

PARAM_INT("knns", "The number of top innner products required",
	  "maxip", 1);
PARAM_INT("leaf_size", "The leaf size for the ball-tree", 
	  "maxip", 20);

PARAM_FLAG("angle_prune", "The flag to trigger the tighter"
	   " pruning using the angles as well", "maxip");
PARAM_FLAG("alt_angle_prune", "The flag to trigger the tighter-er"
	   " pruning using the angles as well", "maxip");
PARAM_FLAG("dual_tree", "The flag to trigger dual-tree "
	   "computation, using a cone tree for the "
	   "queries.", "maxip");

PARAM_FLAG("check_prune", "The flag to trigger the "
	   "checking of the prune.", "maxip");
PARAM_FLAG("alt_dual_traversal", "The flag to trigger the "
	   "alternate dual tree traversal using angles.",
	   "maxip");

#endif
