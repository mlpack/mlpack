/**
 * @file force_error.h
 *
 * @author Jim Waters (jwaters6@gatech.edu)
 *
 * General tree structure for physics problems. Each PhysStat instance
 * represents a particular kind of interaction affecting this particle.
 * This stat houses the kinematic quantities (center of mass, momentum, and 
 * mass) used by all particles for computing dynamics.
 * 
 */

#ifndef FORCE_ERROR_H
#define FORCE_ERROR_H

#include "fastlib/fastlib.h"

struct ForceError { 

  double err_, visited_;
 
  /**
   * Default Initialization
   */
  void Init(){   
  }

  void Init(double err_in, double visited_in){
    err_ = err_in;
    visited_ = visited_in;    
  }

  void Copy(const ForceError* err_in){
    err_ = err_in->err_;
    visited_ = err_in->visited_;
  }

  // Nodes returning from recursion should have same # of visited.
  void Merge(const ForceError& other){
      err_ = min(err_, other.err_);
  }

  int Check(double range, double count){
    int result = 0;
    if (range / count < err_ / visited_){
      result = 1;
    }
    return result;
  }
  
  void AddVisited(double err_in, double count){
    err_ = err_ - err_in;
    visited_ = visited_ - count;
  }

};

#endif



