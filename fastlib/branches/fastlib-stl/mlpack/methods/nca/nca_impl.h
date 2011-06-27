/***
 * @file nca_impl.h
 * @author Ryan Curtin
 *
 * Implementation of templated NCA class.
 */
#ifndef __MLPACK_METHODS_NCA_NCA_IMPL_H
#define __MLPACK_METHODS_NCA_NCA_IMPL_H

// In case it was not already included.
#include "nca.h"

#include <fastlib/optimization/lbfgs/lbfgs.h>
#include "nca_softmax_error_function.h"

namespace mlpack {
namespace nca {

// Just set the internal matrix reference.
template<typename Kernel>
NCA<Kernel>::NCA(const arma::mat& dataset, const arma::uvec& labels) :
    dataset_(dataset), labels_(labels) { /* nothing to do */ }

template<typename Kernel>
void NCA<Kernel>::LearnDistance(arma::mat& output_matrix) {
  output_matrix = arma::eye<arma::mat>(dataset_.n_rows, dataset_.n_rows);

  SoftmaxErrorFunction<Kernel> error_func(dataset_, labels_);

  // We will use the L-BFGS optimizer to optimize the stretching matrix.
  optimization::L_BFGS<SoftmaxErrorFunction<Kernel> > lbfgs(error_func, 10);

  lbfgs.Optimize(0, output_matrix);
}

}; // namespace nca
}; // namespace mlpack

#endif
