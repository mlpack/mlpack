#ifndef TWO_POINT_H
#define TWO_POINT_H

#include "fastlib/fastlib.h"

class TwoPoint{ 

 private:
 
  int n_boxes_;
  Vector counts_, bounds_;
  double max_dist_, min_dist_;

 public:

  OT_DEF(TwoPoint){
    OT_MY_OBJECT(n_boxes_);
    OT_MY_OBJECT(counts_);
    OT_MY_OBJECT(bounds_);   
    OT_MY_OBJECT(max_dist_);
    OT_MY_OBJECT(min_dist_);
  }

 public:
  

  void Init(const Vector& bounds_in){
    n_boxes_ = bounds_in.length()-1;
    bounds_.Init(n_boxes_+1);
    bounds_.CopyValues(bounds_in);    
    counts_.Init(n_boxes_); 
    counts_.SetZero();
    max_dist_ = bounds_[n_boxes_];
  } 

  void Add(double dist, double weight){
    for (int i = 0; i < n_boxes_; i++){
      if (dist >= bounds_[i] && dist < bounds_[i+1]){
	counts_[i] = counts_[i]+weight;
	break;
      }      
    }
  }

  bool InclusionPrune(double min, double max, double count){
    for (int i = 0; i < n_boxes_; i++){
      if (min >= bounds_[i] && min < bounds_[i+1]){
	if(max < bounds_[i+1]){
	  counts_[i] = counts_[i] + count;	 	 
	  return false;	 	  
	} else {
	  return true;
	}
      } 
    }
    return true;
  }

  void Merge(const TwoPoint& other){
    for (int i = 0; i < n_boxes_; i++){
      counts_[i] = counts_[i] + other.counts_[i];
    }
  }

  double Max(){
    return max_dist_;
  }

  double Min(){
    return min_dist_;
  }

  void WriteResult(Vector& counts_out){
    counts_out.Init(n_boxes_);
    counts_out.CopyValues(counts_);
  }

  void Reset(){
    counts_.SetZero();
  }

};

#endif
