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

void IntegralTensor::ContractCoulomb(const ArrayList<index_t>& rho_ind,
                                     const ArrayList<index_t>& sigma_ind, 
                                     const Matrix& density, Matrix* coulomb, 
                                     bool same_ref) {
  
  // sum over index 3 and 4 and contract into a matrix of size 
  // mu_ind.size() * nu_ind.size()
  
  // number of shells used to create the integral array should be equal to the 
  // number of indices in the basis shell
  DEBUG_ASSERT(rho_ind.size() == dim_c_);
  DEBUG_ASSERT(sigma_ind.size() == dim_d_);
  
  //bool same_ref = (rho_ind[0] == sigma_ind[0]) && (rho_ind.back() == sigma_ind.back());

  for (index_t a = 0; a < dim_a_; a++) {
    
    for (index_t b = 0; b < dim_b_; b++) {
      
      double ab_entry = coulomb->ref(a, b);
      
      for (index_t c = 0; c < dim_c_; c++) {
        
        index_t c_ind = rho_ind[c];
        
        for (index_t d = 0; d < dim_d_; d++) {
          
          index_t d_ind = sigma_ind[d];
          
          double this_entry = density.get(c_ind, d_ind) * ref(a, b, c, d);
          
          // takes care of reference symmetry
          if (!same_ref) {
            //printf("accounting for reference symmetry\n");
            this_entry *= 2;
          }
          
          //printf("Coulomb integral(%d, %d, %d, %d): %g\n", a, b, c_ind, d_ind, this_entry);
          
          ab_entry += this_entry;
          
        } // d
      } // c
      
      // take care of query symmetry outside by summing the submatrix into 
      // both sides of the diagonal if necessary
      coulomb->set(a, b, ab_entry);
      
    } // b
  } // a
  
} // ContractCoulomb

void IntegralTensor::ContractExchange(const ArrayList<index_t>& mu_ind,
                                      const ArrayList<index_t>& nu_ind,
                                      const ArrayList<index_t>& rho_ind,
                                      const ArrayList<index_t>& sigma_ind, 
                                      const Matrix& density, Matrix* exchange_ik,
                                      Matrix* exchange_jk, Matrix* exchange_il,
                                      Matrix* exchange_jl) {
  
  // four integrals here
  // does this mean four output matrices
    // don't actually need four if some of the BasisShells are the same
    // but will need to sum over all the integrals even if they are
      // not true - will get to them anyway
  
    
  for (index_t a = 0; a < dim_a_; a++) {
    
    index_t a_ind = mu_ind[a];
    
    for (index_t b = 0; b < dim_b_; b++) {
      
      index_t b_ind = nu_ind[b];
      
      for (index_t c = 0; c < dim_c_; c++) {
        
        index_t c_ind = rho_ind[c];
        
        for (index_t d = 0; d < dim_d_; d++) {
          
          index_t d_ind = sigma_ind[d];
          
          double this_int = ref(a, b, c, d);
          
          double ik_int = density.get(b_ind, d_ind) * this_int;
          
          exchange_ik->set(a, c, exchange_ik->get(a, c) + ik_int);
          
          if (exchange_jk) {
            double jk_int = density.get(a_ind, d_ind) * this_int;
            exchange_jk->set(b, c, exchange_jk->get(b, c) + jk_int);
          }
          
          if (exchange_il) {
            double il_int = density.get(b_ind, c_ind) * this_int;
            exchange_il->set(a, d, exchange_il->get(a, d) + il_int);
          }
          
          if (exchange_jl) {
            double jl_int = density.get(a_ind, c_ind) * this_int;
            exchange_jl->set(b, d, exchange_jl->get(b, d) + jl_int);
          }
            
        } // d
      } // c
    } // b
  } // a
  
  
} // ContractExchange

void IntegralTensor::Print() {
  
  printf("\nIntegrals:\n");
  for (int a = 0; a < dim_a_; a++) {
    for (int b = 0; b < dim_b_; b++) {
      for (int c = 0; c < dim_c_; c++) {
        for (int d = 0; d < dim_d_; d++) {
          
          printf("(%d %d | %d %d) = %g\n", a, b, c, d, ref(a, b, c, d));
          
        }
      }
    }
  }
  printf("\n");
  
} // Print()



void IntegralTensor::UnPermute(int anti_perm) {
 
  /*
  printf("a-perm[0]: %d, a-perm[1]: %d, a-perm[2]: %d, a-perm[3]: %d\n", 
         anti_perm[0], anti_perm[1], anti_perm[2], anti_perm[3]);
  */
  if (anti_perm >= 4) {
    // need to unpermute both
    SwapIndices_(0, 2);
    SwapIndices_(1, 3);
    anti_perm -= 4;
  }
  
  if (anti_perm - 2 >= 0) {
    SwapIndices_(2, 3);
    anti_perm -= 2;
  }
  
  if (anti_perm % 2 == 1) {
    // unpermute the first two
    SwapIndices_(0, 1);
  }

  
  /*
  SwapIndices_(0, anti_perm[0]);
  SwapIndices_(1, anti_perm[1]);
  SwapIndices_(2, anti_perm[2]);
  */
  
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
