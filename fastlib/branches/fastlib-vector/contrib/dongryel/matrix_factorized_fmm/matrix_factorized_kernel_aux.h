/** @file matrix_factorized_kernel_aux.h
 *
 *  @author Dongryeol Lee (dongryel)
 *  @bug No known bugs.
 */

#ifndef MATRIX_FACTORIZED_KERNEL_AUX
#define MATRIX_FACTORIZED_KERNEL_AUX

#include "fastlib/fastlib.h"
#include "matrix_factorized_farfield_expansion.h"
#include "matrix_factorized_local_expansion.h"

class GaussianKernelMatrixFactorizedAux {

 public:
  
  typedef GaussianKernel TKernel;
  
  typedef MatrixFactorizedFarFieldExpansion<GaussianKernelMatrixFactorizedAux> TFarFieldExpansion;

  typedef MatrixFactorizedLocalExpansion<GaussianKernelMatrixFactorizedAux> TLocalExpansion;

  /** @brief The instantiated Gaussian kernel.
   */
  TKernel kernel_;
  
  OT_DEF_BASIC(GaussianKernelMatrixFactorizedAux) {
    OT_MY_OBJECT(kernel_);
  }

 public:

  void Init(double bandwidth, int max_order, int dim) {
    kernel_.Init(bandwidth);
  }

};

class EpanKernelMatrixFactorizedAux {

 public:
  typedef EpanKernel TKernel;
  
  typedef MatrixFactorizedFarFieldExpansion<EpanKernelMatrixFactorizedAux> TFarFieldExpansion;

  typedef MatrixFactorizedLocalExpansion<EpanKernelMatrixFactorizedAux> TLocalExpansion;

  /** @brief The instantiated Epanechnikov kernel.
   */
  TKernel kernel_;
  
  OT_DEF_BASIC(EpanKernelMatrixFactorizedAux) {
    OT_MY_OBJECT(kernel_);
  }

 public:

  void Init(double bandwidth, int max_order, int dim) {
    kernel_.Init(bandwidth);
  }
  
};

#endif
