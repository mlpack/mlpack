#ifndef __MLPACK_CORE_KERNELS_GAUSSIAN_RBF_KERNEL_H
#define __MLPACK_CORE_KERNELS_GAUSSIAN_RBF_KERNEL_H

#include <armadillo>

namespace mlpack {
namespace kernel {

/**
 * The Gaussian radial basis function kernel.
 * This should be documented better.
 */
class GaussianRBFKernel
{
 public:
  /**
   * Default constructor; sets gamma to 0.5.
   */
  GaussianRBFKernel() : gamma(0.5) { }

  /**
   * Given a value for sigma, sets gamma.
   */
  GaussianRBFKernel(double sigma) : gamma(2.0 * pow(sigma, -2.0)) { }

  /**
   * Evaluation of kernel.
   */
  double Evaluate(const arma::vec& a, const arma::vec& b) const {
    arma::vec diff = b - a;
    double distance_squared = arma::dot(diff, diff);
    return exp(gamma * distance_squared);
  }

 private:
  double gamma;
};

}; // namespace kernel
}; // namespace mlpack

#endif
