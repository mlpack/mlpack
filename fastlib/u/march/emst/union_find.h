#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <col/arraylist.h>

/**
 * A Union-Find data structure.  See Cormen, Rivest, & Stein for details.  
 */
class UnionFind {
  
  FORBID_COPY(UnionFind);
  
private:
  
  ArrayList<index_t> parent_;
  ArrayList<int> rank_;
  index_t number_of_elements_;
  
public:
  
  /**
   * Initializes the structure.  This implementation assumes
   * that the size is known advance and fixed
   *
   * @param size The number of elements to be tracked.  
   */
    
    UnionFind() {}
    
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
   */
  index_t Find(index_t x) {
    
    // is this if statement unlikely?
    if (parent_[x] == x) {
      return x; 
    }
    else {
      parent_[x] = Find(parent_[x]);
      return parent_[x];
    }
    
  }
  
  /** 
   * Union the components containing x and y
   * 
   * @param x one component
   * @param y the other component
   */
  void Union(index_t x, index_t y) {
    
    index_t x_root = Find(x);
    index_t y_root = Find(y);
    //DEBUG_ASSERT((x_root != y_root), "x and y are already in the same component, x = %d, y = %d, x_root = %d\n", x, y, x_root);
    
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