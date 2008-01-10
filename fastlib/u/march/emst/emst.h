#ifndef EMST_H
#define EMST_H

//#include "fastlib/fastlib.h"
#include "thor/thor.h"
#include "fastlib/fastlib_int.h"
#include "union_find.h"


const index_t metric = 2;
const index_t no_membership = -1;


class Emst_Stat {
  
private:
  double max_neighbor_distance_;
  index_t component_membership_;
  
public:
    
  void set_max_neighbor_distance(double distance) {
      this->max_neighbor_distance_ = distance;
  }
  double max_neighbor_distance() {
    return max_neighbor_distance_;
  }
  void set_component_membership(index_t membership) {
    this->component_membership_ = membership;
  }
  index_t component_membership() {
    return component_membership_; 
  }
  void Init() {
    set_max_neighbor_distance(DBL_MAX);
    set_component_membership(-1);
  }
  
  void Init(const Matrix& dataset, index_t start, index_t count) {
        
    // Not sure if this is right
    if (count == 1) {
      set_component_membership(start);
      set_max_neighbor_distance(DBL_MAX);
    }
    else {
      Init();
    }
  }
  
  void Init(const Matrix& dataset, index_t start, index_t count,
            const Emst_Stat& left_stat, const Emst_Stat& right_stat) {
    Init(dataset, start, count);
  }
  
}; // class Emst_Stat

typedef BinarySpaceTree<DHrectBound<metric>, Matrix, Emst_Stat> Emst_Tree;

class EdgePair {
  
  //FORBID_ACCIDENTAL_COPIES(EdgePair);
  OT_DEF_BASIC(EdgePair) {
    OT_MY_OBJECT(lesser_index_);
    OT_MY_OBJECT(greater_index_);
    OT_MY_OBJECT(distance_);
  }
  
private:
  index_t lesser_index_;
  index_t greater_index_;
  double distance_;
  
public:
    
    //EdgePair() {}
  
  void Init(index_t lesser, index_t greater, double dist) {
      
    DEBUG_ASSERT_MSG(lesser != greater, "indices equal when creating EdgePair, lesser = %d, distance = %f\n", lesser, dist);
    lesser_index_ = lesser;
    greater_index_ = greater;
    distance_ = dist;
      
  }
  
  index_t lesser_index() {
    return lesser_index_;
  }
  
  index_t greater_index() {
    return greater_index_;
  }
  
  double distance() {
    return distance_;
  }
  
  
  
};// class EdgePair


class DualTreeBoruvka {

  FORBID_ACCIDENTAL_COPIES(DualTreeBoruvka);
  
public:
  
  DualTreeBoruvka() {}
  
  /**
   * @function Init
   * 
   * @param input_tree The kd-tree on the point set
  */
  void Init(Emst_Tree* input_tree) {
    number_of_edges_ = 0;
    tree = input_tree;
    number_of_points = tree->count();
    edges.Init(number_of_points-1);
    connections_.Init(number_of_points);
    total_dist_ = 0.0;
    
    neighbors_in_component.Init(number_of_points);
    neighbors_out_component.Init(number_of_points);
    neighbors_distances.Init(number_of_points);    
    
    for (index_t i = 0; i < number_of_points; i++) {
      
      neighbors_distances[i] = DBL_MAX;
      
    }
    
    VERBOSE_ONLY(ot::Print(neighbors_distances));
    
  }
  
  
  /**
  * @function TestTree
   *
   * Calls a simple recursive test function to see that the tree was initialized properly
   * TODO: Make this a proper unit test
   */
  void TestTree() {
    
    //Test_Tree_Helper_(tree);
    tree->Print();
    
    Emst_Tree* tree_left = tree->left();
    Emst_Tree* tree_right = tree->right();
    
    double bound_dist = tree_left->bound().MinDistanceSq(tree_right->bound());
    printf("bound_dist = %f\n", bound_dist);
    
    
    
    /*tree_left->stat().set_component_membership(1);
    tree_right->stat().set_component_membership(1);
    DEBUG_ASSERT_MSG((tree_left->stat().component_membership() >= 0 
                     && tree_left->stat().component_membership() == tree_right->stat().component_membership()),
                     "component test failed\n");
    printf("component_test succeeded\n");
    */
    /*Emst_Tree* leaf1 = tree;
    Emst_Tree* leaf2 = tree;
    while (leaf1->left() != NULL) {
      leaf1 = leaf1->left(); 
    }
    while (leaf2->right() != NULL) {
      leaf2 = leaf2->right();
    }
    printf("leaf1.count() = %d, leaf2.count() = %d\n", leaf1->count(), leaf2->count());
    printf("distance = %f\n", leaf1->bound().MinDistanceSq(leaf2->bound()));
    */
    
    ComputeNeighbors();
    ot::Print(neighbors_in_component);
    ot::Print(neighbors_out_component);
    ot::Print(neighbors_distances);
    
    
    
  }
  
  index_t get_num_edges() {
    return number_of_edges_;
  }
  
  void add_edge(index_t e1, index_t e2, double distance) {
   
    //EdgePair edge;
    DEBUG_ASSERT_MSG((e1 != e2), "Indices are equal in DualTreeBoruvka.add_edge(%d, %d, %f)\n", e1, e2, distance);
    DEBUG_ASSERT_MSG((distance >= 0.0), "Negative distance input in DualTreeBoruvka.add_edge(%d, %d, %f)\n", e1, e2, distance);
    // Note: I'm not sure that maintaining the order of indices here will be worth this overhead
    if (e1 < e2) {
      edges[number_of_edges_].Init(e1, e2, distance);
    }
    else {
      edges[number_of_edges_].Init(e2, e1, distance);
    }
    
    //edges[number_of_edges_] = edge;
    number_of_edges_++;
    
    
    
  }
  
  /**
  * @function add_all_edges
   *
   * @param all_edges The list of nearest neighbors
   *
   * Adds the edges to the edge list and updates the component information
  */
  void add_all_edges() {
   
    // This function is responsible for adding the entire list of edges
    // IMPORTANT: it must ensure that duplicate edges aren't added - check the UnionFind
    // It is also responsible for updating the UnionFind structure
    // This function must also only add one edge per component - taken care of by component_in
    
    for (index_t i = 0; i < number_of_points; i++) {
     
      index_t component_i = connections_.Find(i);
      index_t in_edge_i = neighbors_in_component[component_i];
      index_t out_edge_i = neighbors_out_component[component_i];
      if (connections_.Find(in_edge_i) != connections_.Find(out_edge_i)) {
        double dist = neighbors_distances[component_i];
        total_dist_ = total_dist_ + dist;
        add_edge(in_edge_i, out_edge_i, dist);
        connections_.Union(in_edge_i, out_edge_i);
      }
      
    }
    
  }
  
  // I have no idea how to refer to the nodes of a kd-tree (or work with one at all)
  // ->bound(), ->stat()
  void compute_neighbors_recursion(Emst_Tree *Q, Emst_Tree *R, double incoming_distance) {
   
    
    if (Q->stat().max_neighbor_distance() < incoming_distance) {
      //pruned by distance
      VERBOSE_MSG(0.0, "distance prune");
    }
    else if (Q->stat().component_membership() >= 0 && Q->stat().component_membership() == R->stat().component_membership()) {
      //pruned by component membership
      VERBOSE_MSG(0.0, "component prune");
    }
    else if (Q->is_leaf() && R->is_leaf()) {
      //base case
      //NOTE: for now, I'm just going to use the distances between the bounding boxes
      // In the future, this is suboptimal.  I will need to add the matrix to this data structure, and use the indices to refer to it
      
      VERBOSE_MSG(0.0, "at base case\n");
      
      // Make sure they're both really leaves
      DEBUG_ASSERT(Q->count() == 1);
      DEBUG_ASSERT(R->count() == 1);
      
      // Make sure it's really a point
      DEBUG_ASSERT(Q->bound().MinDistanceSq(R->bound()) == Q->bound().MaxDistanceSq(R->bound()));
      
      // incoming_distance should be the distance between the two points, so I shouldn't need to compute it
      DEBUG_ASSERT(incoming_distance == Q->bound().MinDistanceSq(R->bound()));
      
      // I think this is likely, because it should have been pruned if it's not
      // Maybe I should make this a DEBUG check
      if (incoming_distance < Q->stat().max_neighbor_distance()) {
        //Q->stat().set_max_neighbor_distance(incoming_distance);
        
        index_t q_point = Q->begin();
        index_t q_component = connections_.Find(q_point);
        DEBUG_ASSERT(q_component == Q->stat().component_membership());
        double component_distance = neighbors_distances[q_component];
        
        if (incoming_distance < component_distance) {
         
          VERBOSE_MSG(0.0, "innermost loop of base case");
          
          index_t r_point = R->begin();
          DEBUG_ASSERT(q_point != r_point);
          DEBUG_ASSERT(connections_.Find(q_point) != connections_.Find(r_point));
          
          neighbors_distances[q_component] = incoming_distance;
          neighbors_in_component[q_component] = q_point;
          neighbors_out_component[q_component] = r_point;
          
        }
        
        
        Q->stat().set_max_neighbor_distance(neighbors_distances[q_component]);
        
        
      }
    }
    // I think this is unlikely, since the recursion is simultaneous on both sides
    else if unlikely(Q->is_leaf()) {
      //recurse on R only 
      VERBOSE_MSG(0.0, "Q is_leaf");
      double left_dist = Q->bound().MinDistanceSq(R->left()->bound());
      double right_dist = Q->bound().MinDistanceSq(R->right()->bound());
      DEBUG_ASSERT(left_dist >= 0.0);
      DEBUG_ASSERT(right_dist >= 0.0);
      
      if (left_dist < right_dist) {
        compute_neighbors_recursion(Q, R->left(), left_dist);
        compute_neighbors_recursion(Q, R->right(), right_dist);
      }
      else {
        compute_neighbors_recursion(Q, R->right(), right_dist);
        compute_neighbors_recursion(Q, R->left(), left_dist);
      }
      
    }
    else if unlikely(R->is_leaf()) {
     //recurse on Q only
      
      VERBOSE_MSG(0.0, "R is_leaf");
      
      double left_dist = Q->left()->bound().MinDistanceSq(R->bound());
      double right_dist = Q->right()->bound().MinDistanceSq(R->bound());
      
      compute_neighbors_recursion(Q->left(), R, left_dist);
      compute_neighbors_recursion(Q->right(), R, right_dist);
      
      Q->stat().set_max_neighbor_distance(max(Q->left()->stat().max_neighbor_distance(), Q->right()->stat().max_neighbor_distance()));
      
    }
    else {
     //recurse on both
      
      VERBOSE_MSG(0.0, "recurse on both");
      
      double left_dist = Q->left()->bound().MinDistanceSq(R->left()->bound());
      double right_dist = Q->left()->bound().MinDistanceSq(R->right()->bound());
      
      if (left_dist < right_dist) {
        compute_neighbors_recursion(Q->left(), R->left(), left_dist);
        compute_neighbors_recursion(Q->left(), R->right(), right_dist);
      }
      else {
        compute_neighbors_recursion(Q->left(), R->right(), right_dist);
        compute_neighbors_recursion(Q->left(), R->left(), left_dist);
      }
      
      left_dist = Q->right()->bound().MinDistanceSq(R->left()->bound());
      right_dist = Q->right()->bound().MinDistanceSq(R->right()->bound());
      
      if (left_dist < right_dist) {
        compute_neighbors_recursion(Q->right(), R->left(), left_dist);
        compute_neighbors_recursion(Q->right(), R->right(), right_dist);
      }
      else {
        compute_neighbors_recursion(Q->right(), R->right(), right_dist);
        compute_neighbors_recursion(Q->right(), R->left(), left_dist);
      }
      
      Q->stat().set_max_neighbor_distance(max(Q->left()->stat().max_neighbor_distance(), Q->right()->stat().max_neighbor_distance()));
      
    }// end else
    
  }
  
  // Makes a call to the recursive portion of the algorithm using the root of the tree
  // IMPORTANT: I need to figure out the best way to keep up with the neighbors found so far
  // I think I should keep an array of index_t: array[i] = j means NN(i) = j
  /**
   * Computes the nearest neighbor of each point in each iteration of the algorithm
   *
   */
  void ComputeNeighbors() {
    
    
    compute_neighbors_recursion(tree, tree, DBL_MAX);
    
  }
  
  // This function handles the while loop and calls the compute neighbors function
  void ComputeMST() {
    
    index_t number_of_loops = 0;
    while (number_of_edges_ < (number_of_points - 1)) {
      ComputeNeighbors();
      // Need to make sure I don't add duplicate edges here
      add_all_edges();
      Cleanup();
      VERBOSE_ONLY(number_of_loops++);
      VERBOSE_ONLY(printf("number_of_loops = %d\n", number_of_loops));
    }
    
    output_results();
    
  }
  
  /* This function resets the values in the nodes of the tree
  * nearest neighbor distance, check for fully connected nodes
  */
  void Cleanup_helper(Emst_Tree* tree) {
    
    tree->stat().set_max_neighbor_distance(DBL_MAX);
    
    if (!tree->is_leaf()) {
      Cleanup_helper(tree->left());
      Cleanup_helper(tree->right());
      
      if (tree->left()->stat().component_membership() >= 0 
          && tree->left()->stat().component_membership() == tree->right()->stat().component_membership()) {
        VERBOSE_MSG(0.0, "connecting components");
        tree->stat().set_component_membership(tree->left()->stat().component_membership());
      }
    }
    else {
      tree->stat().set_component_membership(connections_.Find(tree->begin())); 
    }
    
  }
  
  void Cleanup() {
   
    for (index_t i = 0; i < number_of_points; i++) {
     
      neighbors_distances[i] = DBL_MAX;
      DEBUG_ONLY(neighbors_in_component[i] = BIG_BAD_NUMBER);
      DEBUG_ONLY(neighbors_out_component[i] = BIG_BAD_NUMBER);
      
    }
    
    Cleanup_helper(tree);
    
  }
  
  /* Format and output the results
  */
  void output_results() {
    
    ot::Print(edges);
    // I should make this work with the modules/fast exec stuff
    printf("total squared length = %f\n", total_dist_);
    
  }
  
  
private:
  // What data structure is best for storing the edge list?
  // Is an edge list the best thing to use, or is there a better form?
    
  
  index_t number_of_edges_;
  ArrayList<EdgePair> edges;
  index_t number_of_points;
  UnionFind connections_;
  double total_dist_;
  
  ArrayList<index_t> neighbors_in_component;
  ArrayList<index_t> neighbors_out_component;
  ArrayList<double> neighbors_distances;
  
  Emst_Tree* tree;
  
  // Just does a pre-order depth-first traversal of the tree, printing out the info about each node
 /*( void Test_Tree_Helper_(Emst_Tree* current_tree) {
    
    current_tree->Print();
    
    if (current_tree->left() != NULL) {
      Test_Tree_Helper_(current_tree->left());
      Test_Tree_Helper_(current_tree->right());
    }
    
  }*/
  
}; //class DualTreeBoruvka

#endif
