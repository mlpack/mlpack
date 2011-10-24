/**
 * @file approx_nn_dual.h
 *
 * Defines ApproxNN class to compute the nearest-neighbors 
 * of a query set given a reference set, but obtain the
 * approximate rank nearest neighbor with a given probability.
 */

#ifndef APPROX_NN_DUAL_H
#define APPROX_NN_DUAL_H

#include <fastlib/fastlib.h>
#include <vector>

const fx_entry_doc approx_nn_dual_entries[] = {
  {"dim", FX_PARAM, FX_INT, NULL,
   " The dimension of the data we are dealing with.\n"},
  {"qsize", FX_PARAM, FX_INT, NULL,
   " The number of points in the query set.\n"},
  {"rsize", FX_PARAM, FX_INT, NULL, 
   " The number of points in the reference set.\n"},
  {"knns", FX_PARAM, FX_INT, NULL, 
   " The number of nearest neighbors we need to compute"
   " (defaults to 1).\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   " Rank approximation factor (%% of the reference set size).\n"},
  {"alpha", FX_PARAM, FX_DOUBLE, NULL,
   " The error probability.\n"},
  {"leaf_size", FX_PARAM, FX_INT, NULL,
   " The leaf size for the kd-tree.\n"},
  {"sample_limit", FX_PARAM, FX_INT, NULL,
   " The maximum number of samples"
   " allowed to be made from a single node.\n"},
  {"naive_init", FX_TIMER, FX_CUSTOM, NULL,
   "Naive initialization time.\n"},
  {"naive", FX_TIMER, FX_CUSTOM, NULL,
   "Naive computation time.\n"},
  {"exact_init", FX_TIMER, FX_CUSTOM, NULL,
   "Exact initialization time.\n"},
  {"exact", FX_TIMER, FX_CUSTOM, NULL,
   "Exact computation time.\n"},
  {"approx_init", FX_TIMER, FX_CUSTOM, NULL,
   "Approx initialization time.\n"},
  {"approx", FX_TIMER, FX_CUSTOM, NULL,
   "Approximate computation time.\n"},
  {"tree_building", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to record the time taken to build" 
   " the query and the reference tree.\n"},
  {"tree_building_approx", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to record the time taken to build" 
   " the query and the reference tree for InitApprox.\n"},
  {"computing_sample_sizes", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to compute the sample sizes.\n"},
  {"dual_tree", FX_PARAM, FX_BOOL, NULL,
   "Flag to indicate whether to do single tree or "
   "dual tree (defaults to dual tree = true).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc approx_nn_dual_doc = {
  approx_nn_dual_entries, NULL,
  " Performs approximate nearest neighbors computation"
  " - exact, approximate, brute.\n"
};



/**
 * Performs all-nearest-neighbors.  This class will build the trees and 
 * perform the recursive  computation.
 */
class ApproxNN {
  
  //////////////////////////// Nested Classes /////////////////////////
  /**
   * Extra data for each node in the tree.  For all nearest neighbors, 
   * each node only
   * needs its upper bound on its nearest neighbor distances.  
   */
  class QueryStat {
    OT_DEF_BASIC(QueryStat) {
      OT_MY_OBJECT(max_distance_so_far_); 
      OT_MY_OBJECT(total_points_);
      OT_MY_OBJECT(samples_);
    } // OT_DEF_BASIC
    
  private:
    // The upper bound on the node's nearest neighbor distances.
    double max_distance_so_far_;
    // Number of points considered
    size_t total_points_;
    // Number of points sampled
    size_t samples_;
    
  public:
    // getters
    double max_distance_so_far() {
      return max_distance_so_far_; 
    } 

    size_t total_points() {
      return total_points_;
    }

    size_t samples() {
      return samples_;
    }

    // setters
    void set_max_distance_so_far(double new_dist) {
      max_distance_so_far_ = new_dist; 
    } 

    void set_total_points(size_t points) {
      total_points_ = points;
    }

    void add_total_points(size_t points) {
      total_points_ += points;
    }

    void set_samples(size_t points) {
      samples_ = points;
    }

    void add_samples(size_t points) {
      samples_ += points;
    }
    
    /**
     * Initialization function used in tree-building when initializing 
     * a leaf node.  For allnn, needs no additional information 
     * at the time of tree building.  
     */
    void Init(const Matrix& matrix, size_t start, size_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
      // The points considered starts at zero
      total_points_ = 0;
      // The number of samples starts at zero
      samples_ = 0;
    } 
     
    /**
     * Initialization function used in tree-building when 
     * initializing a non-leaf node.  For other algorithms,
     * node statistics can be built using information from
     * the children.  
     */
    void Init(const Matrix& matrix, size_t start, size_t count, 
	      const QueryStat& left, const QueryStat& right) {
      // For allnn, non-leaves can be initialized in the
      // same way as leaves
      Init(matrix, start, count);
    } 
    
  }; //class QueryStat
  
  // TreeType are BinarySpaceTrees where the data are bounded by 
  // Euclidean bounding boxes, the data are stored in a Matrix, 
  // and each node has a QueryStat for its bound.
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, QueryStat> TreeType;
   
  
  /////////////////////////////// Members ////////////////////////////
private:
  // These will store our data sets.
  Matrix queries_;
  Matrix references_;
  // The pointers to the trees
  TreeType *query_tree_;
  size_t query_;
  std::vector<TreeType*> query_trees_;
  TreeType* reference_tree_;
  // The total number of prunes.
  size_t number_of_prunes_;
  // A permutation of the indices for tree building.
  ArrayList<size_t> old_from_new_queries_;
  ArrayList<size_t> old_from_new_references_;
  // The number of points in a leaf
  size_t leaf_size_;
  // The distance to the candidate nearest neighbor for each query
  Vector neighbor_distances_;
  // The indices of the candidate nearest neighbor for each query
  ArrayList<size_t> neighbor_indices_;

  ArrayList<size_t> nn_dc_;

  // number of nearest neighbrs
  size_t knns_; 
  // The module containing the parameters for this computation. 
  struct datanode* module_;
  // The array containing the sample sizes for the corresponding
  // set sizes
  ArrayList<size_t> sample_sizes_;
  // The rank approximation
  size_t rank_approx_;
  double epsilon_;
  // The maximum number of points to be sampled
  size_t sample_limit_;
  // Minimum number of samples required for each query
  // to maintain the probability bound for the error
  size_t min_samples_per_q_;  
  
  /////////////////////////////// Constructors ////////////////////////
  
  // Add this at the beginning of a class to prevent accidentally
  // calling the copy constructor
  FORBID_ACCIDENTAL_COPIES(ApproxNN);
  
public:
  /**
   * Constructors are generally very simple in FASTlib;
   * most of the work is done by Init().  This is only
   * responsible for ensuring that the object is ready
   * to be destroyed safely.  
   */
  ApproxNN() {
    query_tree_ = NULL;
    query_trees_.clear();
    reference_tree_ = NULL;
  } 
  
  /**
   * The tree is the only member we are responsible for deleting.
   * The others will take care of themselves.  
   */
  ~ApproxNN() {
    if (query_tree_ != NULL) {
      delete query_tree_;
    }
    for (std::vector<TreeType*>::iterator it = query_trees_.begin();
         it < query_trees_.end(); it++) {
      if (*it != NULL) {
        delete *it;
      }
    }
    query_trees_.clear();

    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
  } 
    
  /////////////////////////// Helper Functions //////////////////////
  
private:
  /**
   * Computes the minimum squared distance between the
   * bounding boxes of two nodes
   */
  double MinNodeDistSq_ (TreeType* query_node,
			 TreeType* reference_node) {
    // node->bound() gives us the DHrectBound class for the node
    // It has a function MinDistanceSq which takes another DHrectBound
    return query_node->bound().MinDistanceSq(reference_node->bound());
  } 

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
			   ArrayList<size_t> *samples);

  void ComputeSampleSizes_(size_t rank_approx, double alpha,
			   size_t k, ArrayList<size_t> *samples);


  /**
   * Performs exhaustive computation between two leaves.  
   */
  void ComputeBaseCase_(TreeType* query_node,
			TreeType* reference_node);  
  
  /**
   * The recursive function
   */
  void ComputeNeighborsRecursion_(TreeType* query_node,
				  TreeType* reference_node, 
				  double lower_bound_distance);
  
  /**
   * Performs exhaustive approximate computation
   * between two nodes.
   */
  void ComputeApproxBaseCase_(TreeType* query_node,
			      TreeType* reference_node);

  /**
   * The recursive function for the approximate computation
   */
  void ComputeApproxRecursion_(TreeType* query_node,
			       TreeType* reference_node, 
			       double lower_bound_distance);


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

  // decides whether a query node has enough
  // samples that we can approximate the rest by
  // just picking a small number of samples
  inline bool is_almost_satisfied(TreeType* tree) {
    if (tree->stat().samples() + sample_limit_
	< min_samples_per_q_) {
      return false;
    } else {
      return true;
    }
  }

  // check if the query node has enough samples
  inline bool is_done(TreeType* tree) {
    if (tree->stat().samples() < min_samples_per_q_) {
      return false;
    } else {
      return true;
    }
  }

  void reset_tree_(TreeType *tree);


  /////////////// Public Functions ////////////////////
 public:
  /**
   * Setup the class and build the trees.
   * Note: we are initializing with const references to prevent 
   * local copies of the data.
   */
  void Init(const Matrix& queries_in,
	    const Matrix& references_in,
	    struct datanode* module_in);

  // Initializing for the monochromatic case
  void Init(const Matrix& references_in,
	    struct datanode* module_in);

  void Destruct() {
    if (query_tree_ != NULL) {
      delete query_tree_;
    }
    for (std::vector<TreeType*>::iterator it = query_trees_.begin();
         it < query_trees_.end(); it++) {
      if (*it != NULL) {
        delete *it;
      }
    }
    query_trees_.clear();

    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
    queries_.Destruct();
    references_.Destruct();
    old_from_new_queries_.Renew();
    old_from_new_references_.Renew();
    neighbor_distances_.Destruct();
    neighbor_indices_.Renew();

    sample_sizes_.Renew();

    nn_dc_.Renew();
  }

  /**
   * Initializes the AllNN structure for naive computation.  
   * This means that we simply ignore the tree building.
   */
  void InitNaive(const Matrix& queries_in, 
		 const Matrix& references_in,
		 size_t knns);


  // Initializing for the naive computation for a
  // monochromatic dataset
  void InitNaive(const Matrix& references_in, size_t knns);


  /**
   * Initialization for the Rank-Approximate nearest neighbor
   * computation for which we store the number of samples 
   * to be made for each size of a dataset.
   */
  void InitApprox(const Matrix& queries_in,
		  const Matrix& references_in,
		  struct datanode* module_in);

  // InitApprox for the monochromatic case
  void InitApprox(const Matrix& references_in,
		  struct datanode* module_in);

  // initializing just the sample sizes so that 
  // different values of error can be tried without
  // having to build the tree over and over again.
  void InitSampleSizes();


  // Quick clean up for the trees and the ApproxNN variables
  // after a run of computing neighbors
  void QuickCleanUp();
	
  /**
   * Computes the nearest neighbors and stores them in *results
   */
  void ComputeNeighbors(ArrayList<size_t>* resulting_neighbors,
                        ArrayList<double>* distances);
  
  
  /**
   * Does the entire computation naively
   */
  void ComputeNaive(ArrayList<size_t>* resulting_neighbors,
                    ArrayList<double>*  distances);


  /**
   * Does the entire computation to find the approximate
   * rank NN
   */
  void ComputeApprox(ArrayList<size_t>* resulting_neighbors,
		     ArrayList<double>*  distances);

	
  /**
   * Does the entire computation naively on the sample 
   * (no reference tree).
   */
  void ComputeApproxNoTree(ArrayList<size_t>* resulting_neighbors,
			   ArrayList<double>*  distances);


}; //class AllkNN
#endif
