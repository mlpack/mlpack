/*
 *  integral_tensor.h
 *  
 *
 *  Created by William March on 8/3/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef INTEGRAL_TENSOR_H
#define INTEGRAL_TENSOR_H

#include "fastlib/fastlib.h"

class IntegralTensor {

private:
 
  index_t dim_a_;
  index_t dim_b_;
  index_t dim_c_;
  index_t dim_d_;
  
  Vector ptr_;
  
  index_t num_integrals_;
  
  void Swap_(index_t* i, index_t* j);
  void Swap_(double* i, double* j);
  
  void Swap01_();
  void Swap02_();
  void Swap03_();
  void Swap12_();
  void Swap13_();
  void Swap23_();
  
  // how to make this work for the different indices?
  void SwapIndices_(index_t ind1, index_t ind2);
  
public:
  
  IntegralTensor() {}
  
  ~IntegralTensor() {
  
    //ptr_.Destruct();
    
  }
  
  double* ptr() {
    
    return ptr_.ptr();
    
  }
  
  index_t dim_a() const {
    return dim_a_;
  }

  index_t dim_b() const {
    return dim_b_;
  }

  index_t dim_c() const {
    return dim_c_;
  }

  index_t dim_d() const {
    return dim_d_;
  }
  
  void set(index_t a, index_t b, index_t c, index_t d, double val);
  
  double ref(index_t a, index_t b, index_t c, index_t d);
  
  index_t num_integrals() const {
    
    return num_integrals_;
    
  }
  
  void Init(int num_a, int num_b, int num_c, int num_d, double* integrals);
  
  void UnPermute(const ArrayList<index_t>& anti_perm);
  
}; // class IntegralTensor



#endif