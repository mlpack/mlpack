/**
 * @file raddist.h
 *
 * @author Jim Waters (jwaters6@gatech.edu)
 *
 * General tree structure for physics problems. Each PhysStat instance
 * represents a particular kind of interaction affecting this particle.
 * This stat houses the kinematic quantities (center of mass, momentum, and 
 * mass) used by all particles for computing dynamics.
 * 
 */

#ifndef RADDIST_H
#define RADDIST_H

#include "fastlib/fastlib.h"

class RadDist{ 
 private:
  Vector counts_;
  Vector start_;
  Vector stop_;
  int bins_;

 public:
  /**
   * Default Initialization
   */
  void Init(int bins, double r_max){   
    bins_ = bins;
    counts_.Init(bins);
    start_.Init(bins);
    stop_.Init(bins);
    counts_.SetZero();
    double delta = r_max / bins;
    for (int i = 0; i < bins; i++){
      start_[i] = i*delta;
      stop_[i] = (i+1)*delta;
    }
  }


  void Add(int bin_no, int bin_count){
    counts_[bin_no] = counts_[bin_no] + bin_count;    
  }

  void Add(double value){
    int i = 0;
    if (value < stop_[bins_-1]){
      while (stop_[i] < value){
	i++;
      }    
      Add(i, 1);
    }
  }

  double GetMax(){
    return stop_[bins_-1];
  }

  void Scale(double ratio){
    la::Scale(ratio, &counts_);    
  }

  void Reset(){
    counts_.SetZero();
  }


  void Write(FILE* fp){
    for (int i = 0; i < bins_-1; i++){
      fprintf(fp, "%16.8f, ", counts_[i]);
    }
    fprintf(fp, "%16.8f \n ", counts_[bins_-1]);
  }
  
  void WriteHeader(FILE* fp){
    for (int i = 0; i < bins_-1; i++){
      fprintf(fp, "%16.8f, ", stop_[i]);
    }
    fprintf(fp, "%16.8f \n", stop_[bins_-1]);
  }

};


#endif



