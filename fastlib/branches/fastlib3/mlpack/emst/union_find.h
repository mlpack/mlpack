/**
 * @file union_find.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Implements a union-find data structure.  This structure tracks the components
 * of a graph.  Each point in the graph is initially in its own component.  
 * Calling unionfind.Union(x, y) unites the components indexed by x and y.  
 * unionfind.Find(x) returns the index of the component containing point x.  
 */

#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <fastlib/col/arraylist.h>

/**
 * @class UnionFind
 *
 *A Union-Find data structure.  See Cormen, Rivest, & Stein for details.  
 */
class UnionFind {
  friend class TestUnionFind;
  FORBID_ACCIDENTAL_COPIES(UnionFind);
  
private:
  
  ArrayList<index_t> parent_;
  ArrayList<int> rank_;
  index_t number_of_elements_;
  
public:
  
  UnionFind() {}
  
  ~UnionFind() {}
  
  /**
   * Initializes the structure.  This implementation assumes
   * that the size is known advance and fixed
   *
   * @param size The number of elements to be tracked.  
   */
  
  void Init(index_t size) {
    
    number_of_elements_ = size;
    parent_.Init(number_of_elements_);
    rank_.Init(number_of_elements_);
    for (index_t i = 0; i < number_of_elements_; i++) {
      parent_[i] = i;
      rank_[i] = 0;
    }
    
  }
  
  /**
   * Returns the component containing an element
   *
   * @param x the component to be found
   * @return The index of the component containing x
   */
  index_t Find(index_t x) {
    
    if (parent_[x] == x) {
      return x; 
    }
    else {
      // This ensures that the tree has a small depth
      parent_[x] = Find(parent_[x]);
      return parent_[x];
    }
    
  }
  
  /** 
   * @function Union 
   *
   * Union the components containing x and y
   * 
   * @param x one component
   * @param y the other component
   */
  void Union(index_t x, index_t y) {
    
    index_t x_root = Find(x);
    index_t y_root = Find(y);
    
    if (x_root == y_root) {
      return;    
    }
    else if unlikely(rank_[x_root] == rank_[y_root]) {
      parent_[y_root] = parent_[x_root];
      rank_[x_root] = rank_[x_root] + 1;
    }
    else if (rank_[x_root] > rank_[y_root]) {
      parent_[y_root] = x_root;
    }
    else {
      parent_[x_root] = y_root;
    }
    
  }

  
}; //class UnionFind

#endif
