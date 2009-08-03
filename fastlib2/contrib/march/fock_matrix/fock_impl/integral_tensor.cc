/*
 *  integral_tensor.cc
 *  
 *
 *  Created by William March on 8/3/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "integral_tensor.h"


void IntegralTensor::Swap_(index_t* i, index_t* j) {
  
  index_t temp = *i;
  *i = *j;
  *j = temp;
  
}

void IntegralTensor::Swap_(double* i, double* j) {
  
  double temp = *i;
  *i = *j;
  *j = temp;
  
}

void IntegralTensor::Swap01_() {
  
  Vector temp_ptr;
  temp_ptr.Copy(ptr_);
  
  for (index_t a = 0; a < dim_a_; a++) {
    for (index_t b = 0; b < dim_b_; b++) {
      for (index_t c = 0; c < dim_c_; c++) {
        for (index_t d = 0; d < dim_d_; d++) {
          
          // this does the swap in place, but that isn't correct
          // how to do this efficiently otherwise?
          // don't want to have to index into a temporary array
          
          index_t new_ind = ((b * dim_a_ + a) * dim_c_ + c) * dim_d_ + d;
          temp_ptr[new_ind] = ref(a, b, c, d);
          
        }
      }
    }
  }
  
  Swap_(&dim_a_, &dim_b_);
  ptr_.CopyValues(temp_ptr);
  
} // Swap01_()

void IntegralTensor::Swap02_() {
  
  Vector temp_ptr;
  temp_ptr.Copy(ptr_);
  
  for (index_t a = 0; a < dim_a_; a++) {
    for (index_t b = 0; b < dim_b_; b++) {
      for (index_t c = 0; c < dim_c_; c++) {
        for (index_t d = 0; d < dim_d_; d++) {
          
          // this does the swap in place, but that isn't correct
          // how to do this efficiently otherwise?
          // don't want to have to index into a temporary array
          
          // ((a * dim_b_ + b) * dim_c_ + c) * dim_d_ + d
          index_t new_ind = ((c * dim_b_ + b) * dim_a_ + a) * dim_d_ + d;
          temp_ptr[new_ind] = ref(a, b, c, d);
          
        }
      }
    }
  }
  
  Swap_(&dim_a_, &dim_c_);
  ptr_.CopyValues(temp_ptr);
  
}

void IntegralTensor::Swap03_() {
  
  Vector temp_ptr;
  temp_ptr.Copy(ptr_);
  
  for (index_t a = 0; a < dim_a_; a++) {
    for (index_t b = 0; b < dim_b_; b++) {
      for (index_t c = 0; c < dim_c_; c++) {
        for (index_t d = 0; d < dim_d_; d++) {
          
          // this does the swap in place, but that isn't correct
          // how to do this efficiently otherwise?
          // don't want to have to index into a temporary array
          
          // ((a * dim_b_ + b) * dim_c_ + c) * dim_d_ + d
          index_t new_ind = ((d * dim_b_ + b) * dim_c_ + c) * dim_a_ + a;
          temp_ptr[new_ind] = ref(a, b, c, d);
          
        }
      }
    }
  }
  
  Swap_(&dim_a_, &dim_d_);
  ptr_.CopyValues(temp_ptr);
  
}

void IntegralTensor::Swap12_() {
  
  Vector temp_ptr;
  temp_ptr.Copy(ptr_);
  
  for (index_t a = 0; a < dim_a_; a++) {
    for (index_t b = 0; b < dim_b_; b++) {
      for (index_t c = 0; c < dim_c_; c++) {
        for (index_t d = 0; d < dim_d_; d++) {
          
          // this does the swap in place, but that isn't correct
          // how to do this efficiently otherwise?
          // don't want to have to index into a temporary array
          
          // ((a * dim_b_ + b) * dim_c_ + c) * dim_d_ + d
          index_t new_ind = ((a * dim_c_ + c) * dim_b_ + b) * dim_d_ + d;
          temp_ptr[new_ind] = ref(a, b, c, d);
          
        }
      }
    }
  }
  
  Swap_(&dim_b_, &dim_c_);
  ptr_.CopyValues(temp_ptr);
  
}

void IntegralTensor::Swap13_() {
  
  Vector temp_ptr;
  temp_ptr.Copy(ptr_);
  
  for (index_t a = 0; a < dim_a_; a++) {
    for (index_t b = 0; b < dim_b_; b++) {
      for (index_t c = 0; c < dim_c_; c++) {
        for (index_t d = 0; d < dim_d_; d++) {
          
          // this does the swap in place, but that isn't correct
          // how to do this efficiently otherwise?
          // don't want to have to index into a temporary array
          
          // ((a * dim_b_ + b) * dim_c_ + c) * dim_d_ + d
          index_t new_ind = ((a * dim_d_ + d) * dim_c_ + c) * dim_b_ + b;
          temp_ptr[new_ind] = ref(a, b, c, d);
          
        }
      }
    }
  }
  
  Swap_(&dim_b_, &dim_d_);
  ptr_.CopyValues(temp_ptr);
  
}

void IntegralTensor::Swap23_() {
  
  Vector temp_ptr;
  temp_ptr.Init(ptr_.length());
  
  for (index_t a = 0; a < dim_a_; a++) {
    for (index_t b = 0; b < dim_b_; b++) {
      for (index_t c = 0; c < dim_c_; c++) {
        for (index_t d = 0; d < dim_d_; d++) {
          
          // this does the swap in place, but that isn't correct
          // how to do this efficiently otherwise?
          // don't want to have to index into a temporary array
          
          // ((a * dim_b_ + b) * dim_c_ + c) * dim_d_ + d
          index_t new_ind = ((a * dim_b_ + b) * dim_d_ + d) * dim_c_ + c;
          temp_ptr[new_ind] = ref(a, b, c, d);
          
        }
      }
    }
  }
  
  Swap_(&dim_c_, &dim_d_);
  ptr_.CopyValues(temp_ptr);
  
}

void IntegralTensor::SwapIndices_(index_t ind1, index_t ind2) {

  DEBUG_ASSERT(ind1 >= 0 && ind1 < 4);
  DEBUG_ASSERT(ind2 >= 0 && ind2 < 4);
  
  /*
  if (ind1 == ind2) {
    return;
  }
  
  
  if (ind2 < ind1) {
    Swap_(&ind1, &ind2);
  }
  */
  
  // I think this is correct
  if (ind2 <= ind1) {
    return;
  }
  
  // assume ind2 >= ind1
  
  if (ind1 == 0) {
    
    if (ind2 == 1) {
      
      Swap01_();
      
    }
    else if (ind2 == 2) {
      
      Swap02_();
      
    }
    // ind2 = 3
    else {
      
      Swap03_();
      
    }
    
  }
  else if (ind1 == 1) {
    
    if (ind2 == 2) {
      
      Swap12_();
      
    }
    // ind2 == 3
    else {
     
      Swap13_();
      
    }
    
  }
  else {
    
    // ind1 == 2
    // ind2 == 3
    
    Swap23_();
    
  }
  // ind1 == 3, but then they're the same
} // SwapIndices()
  
void IntegralTensor::set(index_t a, index_t b, index_t c, index_t d, 
                         double val) {
  
  DEBUG_ASSERT(a < dim_a_);
  DEBUG_ASSERT(b < dim_b_);
  DEBUG_ASSERT(c < dim_c_);
  DEBUG_ASSERT(d < dim_d_);
  
  ptr_[((a * dim_b_ + b) * dim_c_ + c) * dim_d_ + d] = val;
  
}

double IntegralTensor::ref(index_t a, index_t b, index_t c, index_t d) {

  DEBUG_ASSERT(a < dim_a_);
  DEBUG_ASSERT(b < dim_b_);
  DEBUG_ASSERT(c < dim_c_);
  DEBUG_ASSERT(d < dim_d_);
  
  return ptr_[((a * dim_b_ + b) * dim_c_ + c) * dim_d_ + d];
  
}


void IntegralTensor::UnPermute(const ArrayList<index_t>& anti_perm) {
 
  printf("a-perm[0]: %d, a-perm[1]: %d, a-perm[2]: %d, a-perm[3]: %d\n", 
         anti_perm[0], anti_perm[1], anti_perm[2], anti_perm[3]);
  
  SwapIndices_(0, anti_perm[0]);
  SwapIndices_(1, anti_perm[1]);
  SwapIndices_(2, anti_perm[2]);
  
}

void IntegralTensor::Init(int num_a, int num_b, int num_c, int num_d, 
                          double* integrals) {
  
  dim_a_ = num_a;
  dim_b_ = num_b;
  dim_c_ = num_c;
  dim_d_ = num_d;
  
  num_integrals_ = dim_a_ * dim_b_ * dim_c_ * dim_d_;
  
  ptr_.Copy(integrals, num_integrals_);
  
}
