#ifndef __MLPACK_CORE_KERNELS_GAUSSIAN_RBF_KERNEL_H
#define __MLPACK_CORE_KERNELS_GAUSSIAN_RBF_KERNEL_H

#include <armadillo>

namespace mlpack {
namespace kernel {

/***
 * Class for Gaussian RBF Kernel
 */
class GaussianRBFKernel
{
  private:
  double gamma;

  public:
  GaussianRBFKernel() { gamma = .5; }
  GaussianRBFKernel(double sigma) {
    gamma = -1.0 / (2 * pow(sigma, 2.0));
  }
  /* Kernel value evaluation */
  double Evaluate(const arma::vec& a, const arma::vec& b) const
  {
    arma::vec diff = b - a;
    double distance_squared = arma::dot(diff, diff);
    return exp(gamma * distance_squared);
  }
};

}; // namespace kernel
}; // namespace mlpack

#endif
