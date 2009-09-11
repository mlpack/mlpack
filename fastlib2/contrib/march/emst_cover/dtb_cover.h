/**
* @file dtb_cover.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Contains an implementation of the DualTreeBoruvka algorithm for finding a 
 * Euclidean Minimum Spanning Tree.  
 */

#ifndef DTB_COVER_H
#define DTB_COVER_H

#include "emst_cover.h"
#include "cover_tree.h"
#include "ctree.h"
#include "mlpack/emst/union_find.h"

/**
* A Stat class for use with fastlib's trees.  This one only stores two values.
 *
 * @param max_neighbor_distance The upper bound on the distance to the nearest 
 * neighbor of any point in this node.
 *
 * @param component_membership The index of the component that all points in 
 * this node belong to.  This is the same index returned by UnionFind for all
 * points in this node.  If points in this node are in different components, 
 * this value will be negative.  
 */
class DTBStat {
  
private:
  
  double max_candidate_distance_;
  index_t component_membership_;
  
public:
    
    void set_max_candidate_distance(double distance) {
      max_candidate_distance_ = distance;
    }
  
  double max_candidate_distance() {
    return max_candidate_distance_;
  }
  
  void set_component_membership(index_t membership) {
    component_membership_ = membership;
  }
  
  index_t component_membership() {
    return component_membership_; 
  }
  
  /** 
    * A generic initializer.
    */
  void Init() {
    
    set_max_neighbor_distance(DBL_MAX);
    set_component_membership(-1);
    
  }
  
  /**
    * An initializer for leaves.
   */
  void Init(const Matrix& dataset, index_t start, index_t count) {
    
    if (count == 1) {
      set_component_membership(start);
      set_max_neighbor_distance(DBL_MAX);
    }
    else {
      Init();
    }
    
  }
  
  /**
    * An initializer for non-leaves.  Simply calls the leaf initializer.
   */
  void Init(const Matrix& dataset, index_t start, index_t count,
            const DTBStat& left_stat, const DTBStat& right_stat) {
    
    Init(dataset, start, count);
    
  }
  
}; // class DTBStat


/**
 * Performs the MST calculation using the Dual-Tree Boruvka algorithm.
 */
class DualCoverTreeBoruvka {

  FORBID_ACCIDENTAL_COPIES(DualCoverTreeBoruvka);
  
 public:
  
  // For now, everything is in Euclidean space
  static const index_t metric = 2;

  //typedef BinarySpaceTree<DHrectBound<metric>, Matrix, DTBStat> DTBTree;
  typdef CoverTreeNode<DTBStat, double> DTBTree;
  
  //////// Member Variables /////////////////////
  
 private:
  
  index_t number_of_edges_;
  ArrayList<EdgePair> edges_;
  index_t number_of_points_;
  UnionFind connections_;
  struct datanode* module_;
  Matrix data_points_;
  index_t leaf_size_;
  
  // lists
  ArrayList<index_t> old_from_new_permutation_;
  ArrayList<index_t> neighbors_in_component_;
  ArrayList<index_t> neighbors_out_component_;
  ArrayList<double> neighbors_distances_;
  
  // output info
  double total_dist_;
  index_t number_of_loops_;
  index_t number_distance_prunes_;
  index_t number_component_prunes_;
  index_t number_leaf_computations_;
  index_t number_q_recursions_;
  index_t number_r_recursions_;
  index_t number_both_recursions_;
  
  int do_naive_;
  
  DTBTree* tree_;
   
  
////////////////// Constructors ////////////////////////
  
 public:

  DualTreeBoruvka() {}
  
  ~DualTreeBoruvka() {
    if (tree_ != NULL) {
      delete tree_; 
    }
  }
  
  
  ////////////////////////// Private Functions ////////////////////
 private:
    
  /**
  * Adds a single edge to the edge list
   */
  void AddEdge_(index_t e1, index_t e2, double distance) {
    
    //EdgePair edge;
    DEBUG_ASSERT_MSG((e1 != e2), 
        "Indices are equal in DualTreeBoruvka.add_edge(%d, %d, %f)\n", 
        e1, e2, distance);
    
    DEBUG_ASSERT_MSG((distance >= 0.0), 
        "Negative distance input in DualTreeBoruvka.add_edge(%d, %d, %f)\n", 
        e1, e2, distance);
    
    if (e1 < e2) {
      edges_[number_of_edges_].Init(e1, e2, distance);
    }
    else {
      edges_[number_of_edges_].Init(e2, e1, distance);
    }
    
    number_of_edges_++;
    
  } // AddEdge_
  
  
  /**
   * Adds all the edges found in one iteration to the list of neighbors.
   */
  void AddAllEdges_() {
    
    for (index_t i = 0; i < number_of_points_; i++) {
      index_t component_i = connections_.Find(i);
      index_t in_edge_i = neighbors_in_component_[component_i];
      index_t out_edge_i = neighbors_out_component_[component_i];
      if (connections_.Find(in_edge_i) != connections_.Find(out_edge_i)) {
        double dist = neighbors_distances_[component_i];
        total_dist_ = total_dist_ + dist;
        AddEdge_(in_edge_i, out_edge_i, dist);
        connections_.Union(in_edge_i, out_edge_i);
      }
    }
    
  } // AddAllEdges_
  
  
  /** 
   * Handles the base case computation.  Also called by naive.
   */
  // TODO: I shouldn't need to loop here, the base case should only be between
  // leaves, which should all have size 1
  // IMPORTANT: assumes that the points are not already connected
  double ComputeBaseCase_(index_t query_point, index_t reference_point) {
    
    DEBUG_ASSERT(connections_.Find(query_point) 
                 != connections_.Find(reference_point));
    
    VERBOSE_MSG(0.0, "at base case\n");
    number_leaf_computations_++;
    
    Vector query_vec;
    Vector ref_vec;
    data_points_.MakeColumnVector(query_point, &query_vec);
    data_points_.MakeColumnVector(reference_point, &ref_vec);
    
    index_t query_component_index = connections_.Find(query_point);
    
    double distance = sqrt(la::DistanceSqEuclidean(query_vec, ref_vec));
    if (distance < neighbors_distances_[query_component_index]) {
      neighbors_distances_[query_component_index] = distance;
      neighbors_in_component_[query_component_index] = query_point;
      neighbors_out_component_[query_component_index] = reference_point;
    }
    
    return neighbors_distances_[query_component_index];
        
  } // ComputeBaseCase_
  
  
  /**
    * Handles the recursive calls to find the nearest neighbors in an iteration
   */
  void ComputeNeighborsRecursion_(DTBTree *query_node, DTBTree *reference_node,
                                  double incoming_distance) {
    
    // Check for a distance prune
    if (query_node->stat().max_neighbor_distance() < incoming_distance) {
      //pruned by distance
      VERBOSE_MSG(0.0, "distance prune");
      number_distance_prunes_++;
    }
    // Check for a component prune
    else if ((query_node->stat().component_membership() >= 0)
             && (query_node->stat().component_membership() == 
                 reference_node->stat().component_membership())) {
      //pruned by component membership
      
      DEBUG_ASSERT(reference_node->stat().component_membership() >= 0);
      
      VERBOSE_MSG(0.0, "component prune");
      number_component_prunes_++;
    }
    // The base case
    else if (query_node->is_leaf() && reference_node->is_leaf()) {
      
      double new_bound = ComputeBaseCase_(query_node->point(),
                                          reference_node->point());
      
      query_node->stat().set_max_neighbor_distance(new_bound);
      
    }
    // Other recursive calls
    else if unlikely(query_node->is_leaf()) {
      //recurse on reference_node only 
      VERBOSE_MSG(0.0, "query_node is_leaf");
      number_r_recursions_++;
      
      // can I just loop over the reference children?
      // I should try to prioritize by distance
      
    }
    else if unlikely(reference_node->is_leaf()) {
      //recurse on query_node only
      
      VERBOSE_MSG(0.0, "reference_node is_leaf");
      number_q_recursions_++;
      
      Vector ref_vec;
      data_points_.MakeColumnVector(reference_node->point());
      
      // just loop over them, prioritizing by distance doesn't matter
      for (index_t i = 0; i < query_node->num_of_children(); i++) {
       
        DTBTree* this_child = query_node->child(i);
        
        Vector child_vec;
        data_points_.MakeColumnVector(this_child->point(), &child_vec);
        
        
        
        double max_dist = sqrt(la::DistanceSqEuclidean(ref_vec, child_vec));
        max_dist += this_child->max_dist_to_grandchild();
        
        ComputeNeighborsRecursion_(this_child, reference_node, max_dist);
        
      } // loop over query children
      
    }
    else {
      //recurse on both
      
      VERBOSE_MSG(0.0, "recurse on both");
      number_both_recursions_++;
      
      
      
    }// end recurse on both
    
  } // ComputeNeighborsRecursion_
  
  /**
    * Computes the nearest neighbor of each point in each iteration 
   * of the algorithm
   */
  void ComputeNeighbors_() {
      ComputeNeighborsRecursion_(tree_, tree_, DBL_MAX);
  } // ComputeNeighbors_
  
  
  /**
    * Unpermute the edge list and output it to results
   *
   * TODO: Make this sort the edge list by distance as well for hierarchical
   * clusterings.
   */
  void EmitResults_(Matrix* results) {
    
    DEBUG_ASSERT(number_of_edges_ == number_of_points_ - 1);
    results->Init(3, number_of_edges_);
    
    if (!do_naive_) {
      for (index_t i = 0; i < (number_of_points_ - 1); i++) {
        
        // unpermute the edge indices
        edges_[i].set_lesser_index(old_from_new_permutation_[edges_[i]
          .lesser_index()]);
        
        edges_[i].set_greater_index(old_from_new_permutation_[edges_[i]
          .greater_index()]);
        
        results->set(0, i, edges_[i].lesser_index());
        results->set(1, i, edges_[i].greater_index());
        results->set(2, i, edges_[i].distance());
        
      }
    }
    else {
      
      for (index_t i = 0; i < number_of_edges_; i++) {
        results->set(0, i, edges_[i].lesser_index());
        results->set(1, i, edges_[i].greater_index());
        results->set(2, i, edges_[i].distance());
      }
      
    }
    
  } // EmitResults_
  
  
  
  /**
    * This function resets the values in the nodes of the tree
   * nearest neighbor distance, check for fully connected nodes
   */
  void CleanupHelper_(DTBTree* tree) {
    
    tree->stat().set_max_neighbor_distance(DBL_MAX);
    
    if (!tree->is_leaf()) {
      CleanupHelper_(tree->left());
      CleanupHelper_(tree->right());
      
      if ((tree->left()->stat().component_membership() >= 0) 
          && (tree->left()->stat().component_membership() == 
              tree->right()->stat().component_membership())) {
        VERBOSE_MSG(0.0, "connecting components");
        tree->stat().set_component_membership(tree->left()->stat().
                                              component_membership());
      }
    }
    else {
      
      index_t new_membership = connections_.Find(tree->begin());
      
      for (index_t i = tree->begin(); i < tree->end(); i++) {
        if (new_membership != connections_.Find(i)) {
          new_membership = -1;
          DEBUG_ASSERT(tree->stat().component_membership() < 0);
          return;
        }
      }
      tree->stat().set_component_membership(new_membership);
      
    }
    
  } // CleanupHelper_
  
  /**
    * The values stored in the tree must be reset on each iteration.  
   */
  void Cleanup_() {
    
    for (index_t i = 0; i < number_of_points_; i++) {
      neighbors_distances_[i] = DBL_MAX;
      DEBUG_ONLY(neighbors_in_component_[i] = BIG_BAD_NUMBER);
      DEBUG_ONLY(neighbors_out_component_[i] = BIG_BAD_NUMBER);
    }
    number_of_loops_++;
    
    if (!do_naive_) {
      CleanupHelper_(tree_);
    }
  }
  
  /**
    * Format and output the results
   */
  void OutputResults_() {
    
    //VERBOSE_ONLY(ot::Print(edges));
    
    fx_format_result(module_, "total_squared_length", "%f", total_dist_);
    fx_format_result(module_, "number_of_points", "%d", number_of_points_);
    fx_format_result(module_, "dimension", "%d", data_points_.n_rows());
    fx_format_result(module_, "number_of_loops", "%d", number_of_loops_);
    fx_format_result(module_, "number_distance_prunes", 
                     "%d", number_distance_prunes_);
    fx_format_result(module_, "number_component_prunes", 
                     "%d", number_component_prunes_);
    fx_format_result(module_, "number_leaf_computations", 
                     "%d", number_leaf_computations_);
    fx_format_result(module_, "number_q_recursions", 
                     "%d", number_q_recursions_);
    fx_format_result(module_, "number_r_recursions", 
                     "%d", number_r_recursions_);
    fx_format_result(module_, "number_both_recursions", 
                     "%d", number_both_recursions_);
    
  } // OutputResults_
  
  /////////// Public Functions ///////////////////
  
 public: 
    
  index_t number_of_edges() {
    return number_of_edges_;
  }

  
  /**
   * Takes in a reference to the data set and a module.  Copies the data, 
   * builds the tree, and initializes all of the member variables.
   *
   * This module will be checked for the optional parameters "leaf_size" and 
   * "do_naive".  
   */
  void Init(const Matrix& data, struct datanode* mod) {
    
    number_of_edges_ = 0;
    data_points_.Copy(data);
    module_ = mod;
    
    do_naive_ = fx_param_exists(module_, "do_naive");
    
    if (!do_naive_) {
      // Default leaf size is 1
      // This gives best pruning empirically
      // Use leaf_size=1 unless space is a big concern
      leaf_size_ = fx_param_int(module_, "leaf_size", 1);
      
      fx_timer_start(module_, "tree_building");

      tree_ = tree::MakeKdTreeMidpoint<DTBTree>
          (data_points_, leaf_size_, &old_from_new_permutation_, NULL);
      
      fx_timer_stop(module_, "tree_building");
    }
    else {
      tree_ = NULL; 
      old_from_new_permutation_.Init(0);
    }
    
    number_of_points_ = data_points_.n_cols();
    edges_.Init(number_of_points_-1);
    connections_.Init(number_of_points_);
    
    neighbors_in_component_.Init(number_of_points_);
    neighbors_out_component_.Init(number_of_points_);
    neighbors_distances_.Init(number_of_points_);    
    
    // Is there a better way to do this?
    // I could use vectors. . .
    for (index_t i = 0; i < number_of_points_; i++) {
      neighbors_distances_[i] = DBL_MAX;
    }
    
    total_dist_ = 0.0;
    number_of_loops_ = 0;
    number_distance_prunes_ = 0;
    number_component_prunes_ = 0;
    number_leaf_computations_ = 0;
    number_q_recursions_ = 0;
    number_r_recursions_ = 0;
    number_both_recursions_ = 0;
    
  } // Init
    
    
  /**
   * Call this function after Init.  It will iteratively find the nearest 
   * neighbor of each component until the MST is complete.
   */
  void ComputeMST(Matrix* results) {
    
    fx_timer_start(module_, "MST_computation");
    
    while (number_of_edges_ < (number_of_points_ - 1)) {
      ComputeNeighbors_();
      
      AddAllEdges_();
      
      Cleanup_();
    
      VERBOSE_ONLY(printf("number_of_loops = %d\n", number_of_loops_));
    }
    
    fx_timer_stop(module_, "MST_computation");
    
    if (results != NULL) {
     
      EmitResults_(results);
      
    }
    
    
    OutputResults_();
    
  } // ComputeMST
  
}; //class DualTreeBoruvka

#endif // inclusion guards
