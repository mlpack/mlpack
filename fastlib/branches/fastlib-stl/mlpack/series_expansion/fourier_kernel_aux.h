#ifndef FOURIER_KERNEL_AUX_H
#define FOURIER_KERNEL_AUX_H

#include "fastlib/fastlib.h"
#include "fourier_series_expansion_aux.h"

template<typename T>
class GaussianKernelFourierAux {
  
 public:
  typedef GaussianKernel TKernel;
  
  typedef FourierSeriesExpansionAux<T> TSeriesExpansionAux;

  /** @brief The Gaussian kernel object.
   */
  TKernel kernel_;
  
  /** @brief The series expansion auxilary object.
   */
  TSeriesExpansionAux sea_;
  
 public:

  void Init(double bandwidth, int max_order, int dim) {
    kernel_.Init(bandwidth);
    sea_.Init(max_order, dim);
  }

  double BandwidthFactor(double bandwidth_sq) const {
    return sqrt(2 * bandwidth_sq);
  }

};

#endif
