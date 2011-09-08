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
 
  size_t dim_a_;
  size_t dim_b_;
  size_t dim_c_;
  size_t dim_d_;
  
  Vector ptr_;
  
  size_t num_integrals_;
  
  void Swap_(size_t* i, size_t* j);
  void Swap_(double* i, double* j);
  
  void Swap01_();
  void Swap02_();
  void Swap03_();
  void Swap12_();
  void Swap13_();
  void Swap23_();
  
  // how to make this work for the different indices?
  void SwapIndices_(size_t ind1, size_t ind2);
  
public:
  
  IntegralTensor() {}
  
  ~IntegralTensor() {
  
    //ptr_.Destruct();
    
  }
  
  double* ptr() {
    
    return ptr_.ptr();
    
  }
  
  size_t dim_a() const {
    return dim_a_;
  }

  size_t dim_b() const {
    return dim_b_;
  }

  size_t dim_c() const {
    return dim_c_;
  }

  size_t dim_d() const {
    return dim_d_;
  }
  
  size_t num_integrals() const {
    return num_integrals_;
  }
  
  void set(size_t a, size_t b, size_t c, size_t d, double val);
  
  double ref(size_t a, size_t b, size_t c, size_t d);
  
  /**
   * The given matrix is assumed to be intitialized and needs to be set to zero
   * before calling this function.  
   */
  void ContractCoulomb(const ArrayList<size_t>& rho_ind,
                       const ArrayList<size_t>& sigma_ind, 
                       const Matrix& density, Matrix* coulomb, bool same_ref);
  
  /**
   * Only specify the pointers needed for the BasisShells
   *
   * Make them NULL if they aren't needed for symmetry
   */
  void ContractExchange(const ArrayList<size_t>& mu_ind,
                        const ArrayList<size_t>& nu_ind,
                        const ArrayList<size_t>& rho_ind,
                        const ArrayList<size_t>& sigma_ind, 
                        const Matrix& density, Matrix* exchange_ik,
                        Matrix* exchange_jk, Matrix* exchange_il,
                        Matrix* exchange_jl);
    
  
  
  void Init(int num_a, int num_b, int num_c, int num_d, double* integrals);
  
  void Print();
  
  void UnPermute(int anti_perm);
  
}; // class IntegralTensor



#endif