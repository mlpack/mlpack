/*
 *  mst_comparison.h
 *  
 *
 *  Created by William March on 10/20/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef MST_COMPARISON_H
#define MST_COMPARISON_H

#include "fastlib/fastlib.h"

class MSTComparison {

private:
  
  Matrix tree1_;
  Matrix tree2_;
  
  
public:
  
  void Init(const Matrix& tree1, const Matrix& tree2) {
    
    tree1_.Copy(tree1);
    tree2_.Copy(tree2);
    
  } // Init()
  
  bool Compare() {
    
    index_t num_edges = tree1_.n_cols();
    
    if (tree2_.n_cols() != num_edges) {
      printf("Trees are not the same size.\n");
      return false; 
    }
    
    //tree1_.PrintDebug("tree1");
    //tree2_.PrintDebug("tree2");
    
    
    
    bool ret_val = true;
    
    for (index_t i = 0; i < num_edges; i++) {
      
      if (tree1_.ref(2, i) - tree2_.ref(2, i) > 0.0001) {
        printf("Distances not equal in position %d.\n", i);
        ret_val = false;
        break;
      }
      
      //if (tree1_.ref(0,i) != tree2_.ref(0,i)) {
      if ((tree1_.ref(0, i) - tree2_.ref(0, i) > 0.0001) &&
          (tree1_.ref(0, i) - tree2_.ref(1, i) > 0.0001)) {
        printf("Lesser index not equal in position %d.\n", i);
        ret_val = false;
        break;
      }
      
      //if (tree1_.ref(1,i) != tree2_.ref(1,i)) {
      if ((tree1_.ref(1, i) - tree2_.ref(1, i) > 0.0001) &&
          (tree1_.ref(1, i) - tree2_.ref(0, i) > 0.0001)) {
        printf("Greater index not equal in position %d.\n", i);
        ret_val = false;
        break;
      }
      
    } // for i
    
    return ret_val;
    
  } // Compare()
  
  
};


#endif