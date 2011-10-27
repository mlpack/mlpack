/**
 * @file approx_max_ip.h
 *
 */

#ifndef APPROX_MAX_IP_H
#define APPROX_MAX_IP_H

#include <assert.h>
#include <mlpack/core.h>
#include <mlpack/core/tree/bounds.h>
#include <mlpack/core/tree/statistic.h>
#include <vector>
#include <armadillo>
#include "general_spacetree.h"
#include "gen_metric_tree.h"
#include "dconebound.h"
#include "gen_cosine_tree.h"

using namespace mlpack;

PARAM_MODULE("approx_maxip", "Parameters for the class that "
	     "builds a tree on the reference set and "
	     "searches for the approximate maximum inner product "
	     "by the branch-and-bound method.");

PARAM_INT("knns", "The number of top innner products required",
	  "approx_maxip", 1);
PARAM_INT("leaf_size", "The leaf size for the ball-tree", 
	  "approx_maxip", 20);

PARAM_DOUBLE("epsilon", "The rank error in terms of the \% of "
	     "reference set size", "approx_maxip", 1.0);
PARAM_DOUBLE("alpha", "The error probability",
	     "approx_maxip", 0.95);
PARAM_INT("sample_limit", "The maximum number of samples allowed "
	  "when the node can be approximated by sampling.", 
	  "approx_maxip", 15);

PARAM_FLAG("angle_prune", "The flag to trigger the tighter"
	   " pruning using the angles as well", "approx_maxip");
PARAM_FLAG("dual_tree", "The flag to trigger dual-tree "
	   "computation, using a cosine tree for the "
	   "queries.", "approx_maxip");

PARAM_FLAG("check_prune", "The flag to trigger the "
	   "checking of the prune.", "approx_maxip");

PARAM_FLAG("no_tree", "The flag to trigger the tree-less "
	   "rank-approximate search.", "approx_maxip");
/**
 * Performs maximum-inner-product-search. 
 * This class will build the trees and 
 * perform the recursive  computation.
 */
class ApproxMaxIP {
  
  //////////////////////////// Nested Classes /////////////////////////
  class QueryStat {
  private:
    double bound_;
    size_t total_points_;
    size_t samples_;

  public:
    double bound() { return bound_; }
    size_t samples() { return samples_; }
    size_t total_points() { return total_points_; }


    void set_bound(double bound) { 
      bound_ = bound;
    }

    void set_total_points(size_t points) { 
      total_points_ = points;
    }

    void set_samples(size_t points) { 
      samples_ = points;
    }

    void add_total_points(size_t points) { 
      total_points_ += points;
    }

    void add_samples(size_t points) { 
      samples_ += points;
    }

    QueryStat() {
      bound_ = 0.0;
      total_points_ = 0;
      samples_ = 0;
    }

    ~QueryStat() {}

    void Init(const arma::mat& data, size_t begin, size_t count) {
      bound_ = 0.0;
      total_points_ = 0;
      samples_ = 0;
    }

    void Init(const arma::mat& data, size_t begin, size_t count,
	      QueryStat& left_stat, QueryStat& right_stat) {
      bound_ = 0.0;
      total_points_ = 0;
      samples_ = 0;
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
  size_t ball_has_origin_;

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


  // Approx search stuff
  arma::Col<size_t> sample_sizes_;
  size_t rank_approx_;
  double epsilon_;

  size_t sample_limit_;
  size_t min_samples_per_q_;

  size_t query_samples_needed_;

  /////////////////////////////// Constructors ////////////////////////
  
public:
  /**
   * Constructors are generally very simple in FASTlib;
   * most of the work is done by Init().  This is only
   * responsible for ensuring that the object is ready
   * to be destroyed safely.  
   */
  ApproxMaxIP() {
    reference_tree_ = NULL;
    query_tree_ = NULL;
  } 
  
  /**
   * The tree is the only member we are responsible for deleting.
   * The others will take care of themselves.  
   */
  ~ApproxMaxIP() {
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
   * This function computes the probability of
   * a particular quantile given the set and sample sizes
   * Computes P(d_(1) <= d_(1+rank_approx))
   */
  double ComputeProbability_(size_t set_size,
                             size_t sample_size,
                             size_t rank_approx);

  /**
   * This function computes the probability of
   * a particular quantile given the set and sample sizes
   * Computes P(d_(k) <= d_(rank_approx))
   */
  double ComputeProbability_(size_t set_size,
                             size_t sample_size, size_t k,
                             size_t rank_approx);
  /**
   * This function computes the minimum sample sizes
   * required to obtain the approximate rank with
   * a given probability (alpha).
   * 
   * It assumes that the ArrayList<size_t> *samples
   * has been initialized to length N.
   */
  void ComputeSampleSizes_(size_t rank_approx, double alpha,
                           arma::Col<size_t> *samples);

  void ComputeSampleSizes_(size_t rank_approx, double alpha,
                           size_t k, arma::Col<size_t> *samples);

  /**
   * Performs exhaustive approximate computation
   * between two nodes.
   */
  void ComputeApproxBaseCase_(TreeType* reference_node);

  void ComputeApproxBaseCase_(CTreeType* query_node,
                              TreeType* reference_node);

  void ComputeBaseCase_(TreeType* reference_node);

  void ComputeBaseCase_(CTreeType* query_node,
			TreeType* reference_node);




  /**
   * The recursive function for the approximate computation
   */
  void ComputeApproxRecursion_(TreeType* reference_node, 
                               double upper_bound_ip);

  void ComputeApproxRecursion_(CTreeType* query_node,
                               TreeType* reference_node, 
                               double upper_bound_ip);

  // decides whether a reference node is small enough
  // to approximate by sampling
  inline bool is_base(TreeType* tree) {
    if (sample_sizes_[tree->end() - tree->begin() -1]
        > sample_limit_) {
      return false;
    } else {
      return true;
    }
  }

  // decides whether a query (node) has enough
  // samples that we can approximate the rest by
  // just picking a small number of samples
  inline bool is_almost_satisfied() {
    if (query_samples_needed_ > sample_limit_) {
      return false;
    } else {
      return true;
    }
  }

  inline bool is_almost_satisfied(CTreeType* tree) {
    if (tree->stat().samples() + sample_limit_
        < min_samples_per_q_) {
      return false;
    } else {
      return true;
    }
  }

  // check if the query (node) has enough samples
  inline bool is_done() {
    if (query_samples_needed_ > 0) {
      return false;
    } else {
      return true;
    }
  }

  inline bool is_done(CTreeType* tree) {
    if (tree->stat().samples() < min_samples_per_q_) {
      return false;
    } else {
      return true;
    }
  }


  void reset_tree_(CTreeType *tree);

  size_t SortValue(double value);

  void InsertNeighbor(size_t pos, size_t point_ind, double value);

  /////////////// Public Functions ////////////////////
public:
  /**
   * Setup the class and build the trees.
   * Note: we are initializing with const references to prevent 
   * local copies of the data.
   */
  void InitApprox(const arma::mat& queries_in, 
		  const arma::mat& references_in);

  void Destruct() {
    if (reference_tree_ != NULL)
      delete reference_tree_;
    
    if (query_tree_ != NULL)
      delete query_tree_;
  }

  /*
   *
   */
  void WarmInitApprox(size_t knns, double epsilon);

  /**
   * Computes the nearest neighbors and stores them in *results
   */

  double ComputeApprox(arma::Mat<size_t>* resulting_neighbors,
		       arma::mat* ips);

  void CheckPrune(CTreeType* query_node, TreeType* ref_node);
  void CheckPrune(TreeType* reference_node);
}; //class ApproxMaxIP

#endif
