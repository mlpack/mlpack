#ifndef THREE_POINT_H
#define THREE_POINT_H

#include "fastlib/fastlib.h"

class ThreePoint{ 

 private:
 
  int n_boxes_;
  ArrayList<ArrayList<Vector> > counts_;
  Vector bounds_;
  double max_dist_, min_dist_;
 
 public:

  OT_DEF(ThreePoint){
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
    for (int i = 0; i < n_boxes_; i++){
      counts_[i].Init(n_boxes_);
      for (int j = 0; j < n_boxes_; j++){
	(counts_[i])[j].Init(n_boxes_);
	(counts_[i])[j].SetZero();
      }
    }
    max_dist_ = bounds_[n_boxes_];
  } 


  void Add(const Vector& dist){
    for (int i = 0; i < n_boxes_; i++){
      if (dist[0] >= bounds_[i] && dist[0] < bounds_[i+1]){
	for (int j = 0; j < n_boxes_; j++){
	  if (dist[1] >= bounds_[j] && dist[1] < bounds_[j+1]){
	    for (int k = 0; k < n_boxes_; k++){
	      if (dist[2] >= bounds_[k] && dist[2] < bounds_[k+1]){
		((counts_[i])[j])[k] = 	((counts_[i])[j])[k] +1;
		break;
	      }  
	    }
	    break;
	  }  
	}     
	break;
      }
    }
  }


  void Merge(const ThreePoint& other){
    for (int i = 0; i < n_boxes_; i++){
      for (int j = 0; j < n_boxes_; j++){
	la::AddTo((other.counts_[i])[j], &((counts_[i])[j]));
      }   
    }
  }

  void Reset(){
    for (int i = 0; i < n_boxes_; i++){
      for (int j = 0; j < n_boxes_; j++){	
	(counts_[i])[j].SetZero();
      }
    }
  }

  double Max(){
    return max_dist_;
  }

  double Min(){
    return min_dist_;
  }

  void WriteResult(ArrayList<ArrayList<Vector> >& counts_out){    
    counts_out.Init(n_boxes_);
    for (int i = 0; i < n_boxes_; i++){
      counts_out[i].Init(n_boxes_);
      for (int j = 0; j < n_boxes_; j++){
	(counts_out[i])[j].Init(n_boxes_);
	(counts_out[i])[j].CopyValues((counts_[i])[j]);
      }   
    }
  }



};

#endif
