/***
 * @file nca_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated NCA class.
 */
#ifndef __MLPACK_METHODS_NCA_NCA_IMPL_HPP
#define __MLPACK_METHODS_NCA_NCA_IMPL_HPP

// In case it was not already included.
#include "nca.hpp"

#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include "nca_softmax_error_function.hpp"

namespace mlpack {
namespace nca {

// Just set the internal matrix reference.
template<typename Kernel>
NCA<Kernel>::NCA(const arma::mat& dataset, const arma::uvec& labels) :
    dataset(dataset), labels(labels) { /* nothing to do */ }

template<typename Kernel>
void NCA<Kernel>::LearnDistance(arma::mat& outputMatrix)
{
  outputMatrix = arma::eye<arma::mat>(dataset.n_rows, dataset.n_rows);

  SoftmaxErrorFunction<Kernel> errorFunc(dataset, labels);

  // We will use the L-BFGS optimizer to optimize the stretching matrix.
  optimization::L_BFGS<SoftmaxErrorFunction<Kernel> > lbfgs(errorFunc, 10);

  Timer::Start("nca_lbfgs_optimization");

  lbfgs.Optimize(outputMatrix);

  Timer::Stop("nca_lbfgs_optimization");
}

}; // namespace nca
}; // namespace mlpack

#endif
