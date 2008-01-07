#ifndef EMST_H
#define EMST_H

//#include "fastlib/fastlib.h"
#include "thor/thor.h"
#include "fastlib/fastlib_int.h"


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
  
private:
  index_t lesser_index;
  index_t greater_index;
  double distance;
  
public:
    void Init(index_t lesser, index_t greater, double dist) {
      
      lesser_index = lesser;
      greater_index = greater;
      distance = dist;
      
    }
  
  
  
};// class EdgePair


class DualTreeBoruvka {

  FORBID_COPY(DualTreeBoruvka);
  
public:
  void Init(Emst_Tree* input_tree) {
    number_of_edges = 0;
    tree = input_tree;
    number_of_points = tree->count();
    edges.Init(number_of_points-1);
  }
  
  
  /**
  * @function TestTree
   *
   * Calls a simple recursive test function to see that the tree was initialized properly
   */
  void TestTree() {
    
    Test_Tree_Helper_(tree);
    
  }
  
  index_t get_num_edges() {
    return number_of_edges;
  }
  
  void add_edge(index_t e1, index_t e2, double distance) {
   
    EdgePair edge;
    DEBUG_ASSERT_MSG((e1 != e2), "Indices are equal in DualTreeBoruvka.add_edge(%d, %d, %f)\n", e1, e2, distance);
    DEBUG_ASSERT_MSG((distance >= 0.0), "Negative distance input in DualTreeBoruvka.add_edge(%d, %d, %f)\n", e1, e2, distance);
    // Note: I'm not sure that maintaining the order of indices here will be worth this overhead
    if (e1 < e2) {
      edge.Init(e1, e2, distance);
    }
    else {
      edge.Init(e2, e1, distance);
    }
    edges[number_of_edges] = edge;
    number_of_edges++;
    
  }
  
  // This function is responsible for adding the entire list of edges
  // IMPORTANT: it must ensure that duplicate edges aren't added
  void add_all_edges(ArrayList<EdgePair>* all_edges) {
   
    
  }
  
  // I have no idea how to refer to the nodes of a kd-tree (or work with one at all)
  void compute_neighbors_recursion(Emst_Tree *Q, Emst_Tree *R) {
   
    
    
  }
  
  // Makes a call to the recursive portion of the algorithm using the root of the tree
  // IMPORTANT: I need to figure out the best way to keep up with the neighbors found so far
  ArrayList<EdgePair>* ComputeNeighbors() {
    
    return NULL;
    
  }
  
  // This function handles the while loop and calls the compute neighbors function
  void ComputeMST() {
    
    while (number_of_edges < number_of_points) {
     
      ArrayList<EdgePair>* these_edges = ComputeNeighbors();
      // Need to make sure I don't add duplicate edges here
      add_all_edges(these_edges);
      Cleanup();
      
    }
    
    output_results();
    
  }
  
  /* This function resets the values in the nodes of the tree
  * nearest neighbor distance, check for fully connected nodes
  */
  void Cleanup() {
    
    
    
  }
  
  /* Format and output the results
  */
  void output_results() {
    
  }
  
  
private:
  // What data structure is best for storing the edge list?
  // Is an edge list the best thing to use, or is there a better form?
    
  
  index_t number_of_edges;
  ArrayList<EdgePair> edges;
  index_t number_of_points;
  
  Emst_Tree* tree;
  
  // Just does a pre-order depth-first traversal of the tree, printing out the info about each node
  void Test_Tree_Helper_(Emst_Tree* current_tree) {
    
    printf("Number of points = %d\n", current_tree->count());
    printf("Begin = %d\n", current_tree->begin());
    printf("End = %d\n", current_tree->end());
    printf("Stat membership = %d\n", current_tree->stat().component_membership());
   // printf("Max distance = %f\n", current_tree->stat().max_neighbor_distance());
    printf("\n");
    
    if (current_tree->left() != NULL) {
      Test_Tree_Helper_(current_tree->left());
      Test_Tree_Helper_(current_tree->right());
    }
    
  }
  
}; //class DualTreeBoruvka

#endif
