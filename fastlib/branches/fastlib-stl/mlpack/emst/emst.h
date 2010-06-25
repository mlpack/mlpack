/** 
* @file emst.h
*
* @author Bill March (march@gatech.edu)
*
* This file contains utilities necessary for all of the minimum spanning tree
* algorithms.
*/

#ifndef EMST_H
#define EMST_H

#include <fastlib/fastlib.h>
#include "union_find.h"

/**
 * An edge pair is simply two indices and a distance.  It is used as the 
 * basic element of an edge list when computing a minimum spanning tree.  
*/
class EdgePair {
  
  //FORBID_ACCIDENTAL_COPIES(EdgePair);
  OT_DEF(EdgePair) {
    OT_MY_OBJECT(lesser_index_);
    OT_MY_OBJECT(greater_index_);
    OT_MY_OBJECT(distance_);
  }
  
private:
  index_t lesser_index_;
  index_t greater_index_;
  double distance_;
  
public:
    
    
    /**
     * Initialize an EdgePair with two indices and a distance.  The indices are
     * called lesser and greater, implying that they be sorted before calling 
     * Init.  However, this is not necessary for functionality; it is just a way
     * to keep the edge list organized in other code.
     */
    void Init(index_t lesser, index_t greater, double dist) {
      
      DEBUG_ASSERT_MSG(lesser != greater, 
          "indices equal when creating EdgePair, lesser = %d, distance = %f\n",
                                                 lesser, dist);
      lesser_index_ = lesser;
      greater_index_ = greater;
      distance_ = dist;
      
    }
  
  index_t lesser_index() {
    return lesser_index_;
  }
  
  void set_lesser_index(index_t index) {
    lesser_index_ = index;
  }
  
  index_t greater_index() {
    return greater_index_;
  }
  
  void set_greater_index(index_t index) {
    greater_index_ = index;
  }
  
  double distance() const {
    return distance_;
  }
    
  void set_distance(double new_dist) {
    distance_ = new_dist; 
  }
  
};// class EdgePair


#endif
