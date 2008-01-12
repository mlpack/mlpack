// inclusion guards, please add them to your .h files
#ifndef ALLNN_H
#define ALLNN_H

// We need to include fastlib.  If you want to use fastlib, you need to have this line in addition to
// the deplibs section of your build.py
#include <fastlib/fastlib.h>

/**
* Performs all-nearest-neighbors.  This class will build the trees.
*/
class AllNN {
  
  //////////////////////////// Nested Classes ///////////////////////////////////////////////
  /**
  * Extra data for each node in the tree.  For all nearest neighbors, each node only
   * needs its upper bound on its nearest neighbor distances.  
   */
  class QueryStat {
    
    // Defines many useful things for a class, including a pretty printer and copy constructor
    OT_DEF_BASIC(QueryStat) {
      // Include this line for all non-pointer members
      // There are other versions for arrays and pointers, see base/otrav.h
      OT_MY_OBJECT(max_distance_so_far_); 
    } // OT_DEF_BASIC
    
private:
    
    /**
    * The upper bound on the node's nearest neighbor distances.
     */
    double max_distance_so_far_;
    
public:
    
    // getter
    double max_distance_so_far() {
      return max_distance_so_far_; 
    } // getter
    
    // setter
    void set_max_distance_so_far(double new_dist) {
      max_distance_so_far_ = new_dist; 
    } // setter
    
    /**
    * Initialization function used in tree-building when initializing a leaf node.  For allnn, 
     * needs no additional information at the time of tree building.  
     */
    void Init(const Matrix& matrix, index_t start, index_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
    } // Init
    
    /**
    * Initialization function used in tree-building when initializing a non-leaf node.
     */
    void Init(const Matrix& matrix, index_t start, index_t count, const QueryStat& left, const QueryStat& right) {
      // For allnn, non-leaves can be initialized the same as leaves
      Init(matrix, start, count);
    } // Init
    
  }; //class AllNNStat  
  
  // QueryTrees are BinarySpaceTrees where the data are bounded by Euclidean bounding boxes,
  // the data are stored in a Matrix, and each node has a QueryStat
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, QueryStat> QueryTree;
  
  // ReferenceTrees are the same as QueryTrees, but don't need node statistics
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> ReferenceTree;  
  
  /////////////////////////////// Members //////////////////////////////////////////////////
private:
  Matrix queries_;
  Matrix references_;
  QueryTree* query_tree_;
  ReferenceTree* reference_tree_;
  index_t number_of_prunes_;
  struct datanode* module_;
  ArrayList<index_t> old_from_new_queries_;
  ArrayList<index_t> old_from_new_references_;
  index_t leaf_size_;
  Vector neighbor_distances_;
  ArrayList<index_t> neighbor_indices_;
  
  
  /////////////////////////////// Constructors /////////////////////////////////////////////
  
  // Add this at the beginning of a class to prevent accidentally calling the copy constructor
  FORBID_ACCIDENTAL_COPIES(AllNN);
  

public:

  /**
  * Constructors are generally very simple in FASTlib; most of the work is done by Init().  It is only
  * responsible for ensuring that the object is ready to be destroyed safely.  
  */
  AllNN() {
    query_tree_ = NULL;
    reference_tree_ = NULL;
  } // AllNN
  
  /**
  * The tree is the only member we are responsible for deleting.  The others will take care of themselves.  
  */
  ~AllNN() {
    if (query_tree_ != NULL) {
      delete query_tree_;
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
  } // ~AllNN
    
      
 /////////////////////////////// Helper Functions ///////////////////////////////////////////////////
  
  /**
  */
  double MinNodeDistSq_ (QueryTree* query_node, ReferenceTree* reference_node) {
   
    // Explain this
    return query_node->bound().MinDistanceSq(reference_node->bound());
    
  } // MinNodeDist_
  
  
  /**
  */
  void ComputeBaseCase_(QueryTree* query_node, ReferenceTree* reference_node) {
   
    
    double query_max_neighbor_distance = -1.0;
    
    // Explain this
    for (index_t query_index = query_node->begin(); query_index < query_node->end(); query_index++) {
      
      // Explain this
      Vector query_point;
      queries_.MakeColumnVector(query_index, &query_point);
      
      for (index_t reference_index = reference_node->begin(); reference_index < reference_node->end(); reference_index++) {
        
        Vector reference_point;
        references_.MakeColumnVector(reference_index, &reference_point);
        
        // Explain this
        double distance = la::DistanceSqEuclidean(query_point, reference_point);
        
        // Explain this
        if (distance < neighbor_distances_[query_index]) {
          neighbor_distances_[query_index] = distance;
          neighbor_indices_[query_index] = reference_index;
        }
        
        // Explain this
        if (neighbor_distances_[query_index] > query_max_neighbor_distance) {
          query_max_neighbor_distance = neighbor_distances_[query_index]; 
        }
        
      } // for reference_index
      
    } // for query_index
    
    query_node->stat().set_max_distance_so_far(query_max_neighbor_distance);
         
    
  } // ComputeBaseCase_
  
  
  /**
  */
  void ComputeNeighborsRecursion_ (QueryTree* query_node, ReferenceTree* reference_node) {
   
    // Explain this
    DEBUG_ASSERT(query_node != NULL);
    DEBUG_ASSERT_MSG(reference_node != NULL, "reference node is null");
    
    // Explain this (could have passed it)
    double lower_bound_distance = MinNodeDistSq_(query_node, reference_node);
    
    if (lower_bound_distance > query_node->stat().max_distance_so_far()) {
      // Pruned by distance
      number_of_prunes_++;
    }
    else if (query_node->is_leaf() && reference_node->is_leaf()) {
      // Base Case
      ComputeBaseCase_(query_node, reference_node);
    }
    else if (query_node->is_leaf()) {
      // Only query is a leaf
      
      // Explain this 
      double left_distance = MinNodeDistSq_(query_node, reference_node->left());
      double right_distance = MinNodeDistSq_(query_node, reference_node->right());
      
      if (left_distance < right_distance) {
        ComputeNeighborsRecursion_(query_node, reference_node->left());
        ComputeNeighborsRecursion_(query_node, reference_node->right());
      }
      else {
        ComputeNeighborsRecursion_(query_node, reference_node->right());
        ComputeNeighborsRecursion_(query_node, reference_node->left());
      }
      
    }
    
    else if (reference_node->is_leaf()) {
      // Only reference is a leaf 
      ComputeNeighborsRecursion_(query_node->left(), reference_node);
      ComputeNeighborsRecursion_(query_node->right(), reference_node);
      
      // Explain this
      query_node->stat().set_max_distance_so_far(
          max(query_node->left()->stat().max_distance_so_far(),
              query_node->right()->stat().max_distance_so_far()));
      
    }
    
    else {
      // Recurse on both
      
      double left_distance = MinNodeDistSq_(query_node->left(), reference_node->left());
      double right_distance = MinNodeDistSq_(query_node->left(), reference_node->right());
      
      if (left_distance < right_distance) {
        ComputeNeighborsRecursion_(query_node->left(), reference_node->left());
        ComputeNeighborsRecursion_(query_node->left(), reference_node->right());
      }
      else {
        ComputeNeighborsRecursion_(query_node->left(), reference_node->right());
        ComputeNeighborsRecursion_(query_node->left(), reference_node->left());
      }
      
      left_distance = MinNodeDistSq_(query_node->right(), reference_node->left());
      right_distance = MinNodeDistSq_(query_node->right(), reference_node->right());
      
      if (left_distance < right_distance) {
        ComputeNeighborsRecursion_(query_node->right(), reference_node->left());
        ComputeNeighborsRecursion_(query_node->right(), reference_node->right());
      }
      else {
        ComputeNeighborsRecursion_(query_node->right(), reference_node->right());
        ComputeNeighborsRecursion_(query_node->right(), reference_node->left());
      }
      
      query_node->stat().set_max_distance_so_far(
          max(query_node->left()->stat().max_distance_so_far(),
              query_node->right()->stat().max_distance_so_far()));
      
    }
    
  } // ComputeNeighborsRecursion_
  
  
  
  
  ////////////////////////////////// Public Functions ////////////////////////////////////////////////
  
  /**
  * Setup the class and build the trees.  Note: we are initializing with const references to prevent 
  * local copies.
  */
  void Init(const Matrix& queries_in, const Matrix& references_in, struct datanode* module_in) {
    
    module_ = module_in;
    
    // Get the leaf size
    leaf_size_ = fx_param_int(module_, "leaf_size", 20);
    // Explain this
    DEBUG_ASSERT(leaf_size_ > 0);
    
    // Copy the matrices to the class members since they will be rearranged.  
    queries_.Copy(queries_in);
    references_.Copy(references_in);
    
    // Explain this
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
    // Explain this
    neighbor_indices_.Init(queries_.n_cols());
    
    // Explain this
    neighbor_distances_.Init(queries_.n_cols());
    neighbor_distances_.SetAll(DBL_MAX);
    
    // Explain this
    // Instead of NULL, it is possible to specify an array new_from_old_
    query_tree_ = tree::MakeKdTreeMidpoint<QueryTree>(queries_, leaf_size_, &old_from_new_queries_, NULL);
    reference_tree_ = tree::MakeKdTreeMidpoint<ReferenceTree>(references_, leaf_size_, &old_from_new_references_, NULL);
    
  } // Init
  
  
  /**
  * 
  */
  void ComputeNeighbors(ArrayList<index_t>* results) {
    
    ComputeNeighborsRecursion_(query_tree_, reference_tree_);
    
    // Explain this (for real)
    results->Init(neighbor_indices_.size());
    for (index_t i = 0; i < neighbor_indices_.size(); i++) {
      
      (*results)[old_from_new_queries_[i]] = old_from_new_references_[neighbor_indices_[i]];
      
    } // for i
    
    
    
  } // ComputeNeighbors
  
  
  void ComputeNaive(ArrayList<index_t>* results) {
    
    ComputeBaseCase_(query_tree_, reference_tree_);
    
    // Explain this (for real)
    results->Init(neighbor_indices_.size());
    for (index_t i = 0; i < neighbor_indices_.size(); i++) {
      
      (*results)[old_from_new_queries_[i]] = old_from_new_references_[neighbor_indices_[i]];
      
    } // for i
    
  } // ComputeNaive
  
  
   
}; //class AllNN




#endif
// end inclusion guards