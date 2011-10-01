/**
 * @file time_constrained_nn.h
 *
 * Defines TCNN class to perform all-nearest-neighbors on the specified 
 * data set, but returns the rank error obtained if the search is time 
 * constrained. 
 */

#ifndef TIME_CONSTRAINED_NN_H
#define TIME_CONSTRAINED_NN_H

#include <fastlib/fastlib.h>
#include <vector>
#include <string>

const fx_entry_doc time_constrained_nn_entries[] = {
  {"dim", FX_PARAM, FX_INT, NULL,
   " The dimension of the data we are dealing with.\n"},
  {"qsize", FX_PARAM, FX_INT, NULL,
   " The number of points in the training query set.\n"},
  {"rsize", FX_PARAM, FX_INT, NULL, 
   " The number of points in the reference set.\n"},
  {"knns", FX_PARAM, FX_INT, NULL, 
   " The number of nearest neighbors we need to compute"
   " (defaults to 1).\n"},
  {"leaf_size", FX_PARAM, FX_INT, NULL,
   " The leaf size for the kd-tree.\n"},
  {"init", FX_TIMER, FX_CUSTOM, NULL,
   "Exact initialization time.\n"},
  {"exact", FX_TIMER, FX_CUSTOM, NULL,
   "Exact computation time.\n"},
  {"tree_building", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to record the time taken to build" 
   " the query and the reference tree.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc time_constrained_nn_doc = {
  time_constrained_nn_entries, NULL,
  " Performs nearest neighbors computation"
  " - exact, time constrained.\n"
};


/**
 * Performs all-nearest-neighbors.  This class will build the trees and 
 * perform the recursive  computation.
 */
class TCNN {
  
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
    void Init(const Matrix& matrix, size_t start, size_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
    } 
     
    /**
     * Initialization function used in tree-building when 
     * initializing a non-leaf node.  For other algorithms,
     * node statistics can be built using information from the children.  
     */
    void Init(const Matrix& matrix, size_t start, size_t count, 
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

  // This will store the query index for the single tree run
  size_t query_;

  // Pointers to the roots of the two trees.
  std::vector<TreeType*> query_trees_;
  TreeType* reference_tree_;

  // the rank matrix for the queries on this reference set
  Matrix rank_matrix_;
  GenVector<size_t> rank_vec_;
  bool rank_file_too_big_;
  FILE *rank_fp_;

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

//   Vector ann_dist_;
//   ArrayList<size_t> ann_ind_;

  ArrayList<size_t> nn_mc_;
  ArrayList<size_t> nn_dc_;

  // setting up the list of errors for subsequent number of leaves
  std::vector<std::vector<size_t>* > *error_list_;
  std::vector<std::vector<size_t>* > *nn_dc_list_;

  size_t number_of_leaves_;

  // number of nearest neighbrs
  size_t knns_; 
  // The module containing the parameters for this computation. 
  struct datanode* module_;
  /////////////////////////////// Constructors ////////////////////////
  
  // Add this at the beginning of a class to prevent accidentally
  // calling the copy constructor
  FORBID_ACCIDENTAL_COPIES(TCNN);
  
public:
  /**
   * Constructors are generally very simple in FASTlib;
   * most of the work is done by Init().  This is only
   * responsible for ensuring that the object is ready
   * to be destroyed safely.  
   */
  TCNN() {
    reference_tree_ = NULL;
    query_trees_.clear();
  } 
  
  /**
   * The tree is the only member we are responsible for deleting.
   * The others will take care of themselves.  
   */
  ~TCNN() {
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
				  double lower_bound_distance);
  

  /////////////// Public Functions ////////////////////
public:
  /**
   * Setup the class and build the trees.
   * Note: we are initializing with const references to prevent 
   * local copies of the data.
   */

  void Destruct() {
    for (std::vector<TreeType*>::iterator it
	   = query_trees_.begin();
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

    nn_dc_.Renew();
    nn_mc_.Renew();

    delete error_list_;
    delete nn_dc_list_;
  }

  void Init(const Matrix& references,
	    struct datanode* module);

  void InitQueries(const Matrix& queries,
		   const Matrix& rank_matrix);

  void InitQueries(const Matrix& queries,
		   const std::string rank_matrix_file);

  void ComputeNeighborsSequential(ArrayList<double> *means,
				  ArrayList<double> *stds,
				  ArrayList<size_t> *maxs,
				  ArrayList<size_t> *mins);  
}; //class TCNN


#endif

