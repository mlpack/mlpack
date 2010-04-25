/*
 *  n_point_multi.h
 *  
 *
 *  Created by William March on 4/14/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef N_POINT_MULTI_H
#define N_POINT_MULTI_H

#include "multi_matcher.h"
#include "fastlib/fastlib.h"
#include "n_point_impl.h"

class NPointMulti { 
  
private:
  
  class NPointStat {
    
  private:
    
    index_t node_index_;
    
  public:
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      node_index_ = -1;
      
    } // Init() leaves
    
    void Init(const Matrix& dataset, index_t start, index_t count,
              const NPointStat& left_stat, const NPointStat& right_stat) {
      
      node_index_ = -1;
      
    } // Init() non-leaves
    
    index_t node_index() const {
      return node_index_; 
    }
    
    void set_node_index(index_t ind) {
      node_index_ = ind;
    }
    
  }; // NPointStat
  
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, NPointStat> NPointNode;
  
/////////////////////
  
  int num_bandwidths_;
  
  MultiMatcher matcher_;
  
  
  
  
  
  
  ///////////////// functions ////////////////////
  
  
  void CheckNodeList_(ArrayList<NPointNode*>& nodes);
  
  
  
  
}; // NPointMulti


#endif
