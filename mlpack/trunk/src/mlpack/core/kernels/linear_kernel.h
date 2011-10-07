#ifndef __MLPACK_CORE_KERNELS_LINEAR_KERNEL_H
#define __MLPACK_CORE_KERNELS_LINEAR_KERNEL_H

#include <armadillo>

namespace mlpack {
namespace kernel {

/**
* Class for Linear Kernel
*/
class LinearKernel
{
  public:
  LinearKernel() {}

  /* Kernel value evaluation */
  double Evaluate(const arma::vec& a, const arma::vec& b) const
  {
    return arma::dot(a, b);
  }
  /* Kernel name */
  void GetName(std::string& kname) {
    kname = "linear";
  }
  /* Get an type ID for kernel */
  size_t GetTypeId()
  {
    return 0;
  }
};

}; //namespace kernel
}; //namespace mlpack

#endif
