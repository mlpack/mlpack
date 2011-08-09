/**
 * @file approx_nn.h
 *
 * Defines ApproxNN class to perform all-nearest-neighbors on two specified 
 * data sets, but obtain the approximate rank nearest neighbor with a 
 * given probability.
 */

#ifndef APPROX_NN_H
#define APPROX_NN_H

#include <fastlib/fastlib.h>
#include <vector>
#include <string>

const fx_entry_doc approx_nn_entries[] = {
  {"dim", FX_PARAM, FX_INT, NULL,
   " The dimension of the data we are dealing with.\n"},
  {"qsize", FX_PARAM, FX_INT, NULL,
   " The number of points in the training query set.\n"},
  {"test_qsize", FX_PARAM, FX_INT, NULL,
   " The number of points in the test query set.\n"},
  {"rsize", FX_PARAM, FX_INT, NULL, 
   " The number of points in the reference set.\n"},
  {"knns", FX_PARAM, FX_INT, NULL, 
   " The number of nearest neighbors we need to compute"
   " (defaults to 1).\n"},
  //  {"epsilon", FX_PARAM, FX_INT, NULL,
  // " Rank approximation.\n"},
  {"dist_epsilon", FX_PARAM, FX_DOUBLE, NULL,
   " Dist approximation factor.\n"},
  {"alpha", FX_PARAM, FX_DOUBLE, NULL,
   " The error probability.\n"},
  {"leaf_size", FX_PARAM, FX_INT, NULL,
   " The leaf size for the kd-tree.\n"},
//   {"sample_limit", FX_PARAM, FX_INT, NULL,
//    " The maximum number of samples"
//    " allowed to be made from a single node.\n"},
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
//   {"tree_building_approx", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to record the time taken to build" 
//    " the query and the reference tree for InitApprox.\n"},
//   {"computing_sample_sizes", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to compute the sample sizes.\n"},
  {"e_v_dc_file", FX_PARAM, FX_STR, NULL,
   " A file where error, avg. dc, avg. mc values"
   " would be written into.\n"},
  {"uq_vq_file", FX_PARAM, FX_STR, NULL,
   " A file where the u_q v_q values would be written.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc approx_nn_doc = {
  approx_nn_entries, NULL,
  " Performs nearest neighbors computation"
  " - exact, approximate.\n"
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
    
    // Defines many useful things for a class, including a pretty 
    // printer and copy constructor
    OT_DEF_BASIC(QueryStat) {
      // Include this line for all non-pointer members
      // There are other versions for arrays and pointers, see base/otrav.h
      OT_MY_OBJECT(max_distance_so_far_); 
    } // OT_DEF_BASIC
    
  private:
    // The upper bound on the node's nearest neighbor distances.
    double max_distance_so_far_;
    
  public:
    // getters
    double max_distance_so_far() {
      return max_distance_so_far_; 
    } 

    // setters
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
    void Init(const Matrix& matrix, index_t start, index_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
    } 
     
    /**
     * Initialization function used in tree-building when 
     * initializing a non-leaf node.  For other algorithms,
     * node statistics can be built using information from the children.  
     */
    void Init(const Matrix& matrix, index_t start, index_t count, 
	      const QueryStat& left, const QueryStat& right) {
      // For allnn, non-leaves can be initialized in the same way as leaves
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

  // Matrix test_queries_;

  // This will store the query index for the single tree run
  index_t query_;

  // index_t test_query_;

  // Pointers to the roots of the two trees.
  std::vector<TreeType*> query_trees_;
  TreeType* reference_tree_;

  // std::vector<TreeType*> test_query_trees_;

  // The total number of prunes.
  index_t number_of_prunes_;
  // A permutation of the indices for tree building.
  ArrayList<index_t> old_from_new_queries_;
  ArrayList<index_t> old_from_new_references_;

//   ArrayList<index_t> old_from_new_test_queries_;
  // The number of points in a leaf
  index_t leaf_size_;

  // The distance to the candidate nearest neighbor for each query
  Vector neighbor_distances_;
  // The indices of the candidate nearest neighbor for each query
  ArrayList<index_t> neighbor_indices_;

//   Vector ann_dist_;
//   ArrayList<index_t> ann_ind_;

  ArrayList<index_t> nn_mc_;
  ArrayList<index_t> nn_dc_;

  // setting up the list of errors for subsequent number of leaves
  std::vector<index_t> *error_list_;

  std::vector<double> *dist_error_list_;
  std::vector<double> *sq_dist_error_list_;
  std::vector<double> *max_dist_error_list_;

  // std::vector<index_t> *rank_error_list_;
  std::vector<long int> *ann_mc_;

  std::vector<long int> *ann_dc_;
  std::vector<long int> *sq_ann_dc_;
  std::vector<long int> *max_ann_dc_;

  std::vector<int> *u_q_;
  std::vector<int> *v_q_;

  index_t compute_nn_;
  index_t number_of_leaves_;

  Vector calc_nn_dists_;

  long int last_dc_val_sum_;
  long int last_sq_dc_val_sum_;

  long int last_mc_val_sum_;


  // setting up the stats for the test queries
  index_t train_nn_;
  index_t test_ann_;
  index_t max_leaves_;

  ArrayList<index_t> test_nn_mc_;
  ArrayList<index_t> test_nn_dc_;

  ArrayList<index_t> test_ann_mc_;
  ArrayList<index_t> test_ann_dc_;

  // user specified error bounds
  double dist_epsilon_;
  double alpha_;


  // number of nearest neighbrs
  index_t knns_; 
  // The module containing the parameters for this computation. 
  struct datanode* module_;
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
    reference_tree_ = NULL;
    query_trees_.clear();
  } 
  
  /**
   * The tree is the only member we are responsible for deleting.
   * The others will take care of themselves.  
   */
  ~ApproxNN() {
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
  double MinNodeDistSq_ (TreeType* query_node, TreeType* reference_node) {
    return query_node->bound().MinDistanceSq(reference_node->bound());
  } 

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
				  double lower_bound_distance) {

    DEBUG_ASSERT(query_node != NULL);
    DEBUG_ASSERT(reference_node != NULL);

    DEBUG_ASSERT(lower_bound_distance
		 == MinNodeDistSq_(query_node, reference_node));

    // just checking for the single tree version
    DEBUG_ASSERT(query_node->end()
		 - query_node->begin() == 1);
    DEBUG_ASSERT(query_node->is_leaf());

//     if (compute_nn_ == 1 || number_of_leaves_ > 0) {
    
      if (lower_bound_distance > query_node->stat().max_distance_so_far()) {
	// Pruned by distance
	number_of_prunes_++;
      }
      // node->is_leaf() works as one would expect
      else if (query_node->is_leaf() && reference_node->is_leaf()) {
	// Base Case
	ComputeBaseCase_(query_node, reference_node);
      }
      else if (query_node->is_leaf()) {
	// Only query is a leaf
      
	// incrementing the number of margin computations
	nn_mc_[query_]++;

	// We'll order the computation by distance 
	double left_distance = MinNodeDistSq_(query_node,
					      reference_node->left());
	double right_distance = MinNodeDistSq_(query_node,
					       reference_node->right());
      
	if (left_distance < right_distance) {
	  ComputeNeighborsRecursion_(query_node, reference_node->left(), 
				     left_distance);
	  ComputeNeighborsRecursion_(query_node, reference_node->right(), 
				     right_distance);
	}
	else {
	  ComputeNeighborsRecursion_(query_node, reference_node->right(), 
				     right_distance);
	  ComputeNeighborsRecursion_(query_node, reference_node->left(), 
				     left_distance);
	}

      }
//     } // to compute nn or any mopre leaves any more
  } // ComputeNeighborsRecursion_
  

  /////////////// Public Functions ////////////////////
public:
  /**
   * Setup the class and build the trees.
   * Note: we are initializing with const references to prevent 
   * local copies of the data.
   */
  void InitTrain(const Matrix& queries_in,
		 const Matrix& references_in,
		 struct datanode* module_in);

  void InitTest(const Matrix& queries_in);

  void Destruct() {
    for (std::vector<TreeType*>::iterator it = query_trees_.begin();
	 it < query_trees_.end(); it++) {
      if (*it != NULL) {
	delete *it;
      }
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
    queries_.Destruct();
    references_.Destruct();
    old_from_new_queries_.Renew();
    old_from_new_references_.Renew();
    neighbor_distances_.Destruct();
    neighbor_indices_.Renew();

    calc_nn_dists_.Destruct();

//     ann_ind_.Renew();
//     ann_dist_.Destruct();

    nn_dc_.Renew();
    nn_mc_.Renew();

//     ann_dc_.Renew();
//     ann_mc_.Renew();
    delete error_list_;
    delete dist_error_list_;
    delete sq_dist_error_list_;
    delete max_dist_error_list_;
    delete ann_mc_;
    delete ann_dc_;
    delete sq_ann_dc_;
    delete max_ann_dc_;

    delete u_q_;
    delete v_q_;
  }

  /**
   * Trains the nearest neighbors data structure
   * and computes estimates of the minimum number 
   * of leaves required to traverse to compute nn
   * within specified bounds
   */
  void TrainNeighbors();
 
  void TestNeighbors(ArrayList<index_t>* nn_ind,
		     ArrayList<double>* nn_dist, 
		     ArrayList<index_t>* ann_ind,
		     ArrayList<double>* ann_dist);

  
}; //class ApproxNN


#endif

