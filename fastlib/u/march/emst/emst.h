#ifndef EMST_H
#define EMST_H

#include "fastlib/fastlib.h"
#include "union_find.h"

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
  
  double distance() {
    return distance_;
  }
  
};// class EdgePair


#endif
