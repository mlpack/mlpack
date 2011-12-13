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
template<typename MetricType, typename MatType>
NCA<MetricType, MatType>::NCA(const MatType& dataset,
                              const arma::uvec& labels) :
    dataset_(dataset),
    labels_(labels)
{
  /* nothing to do */
}

template<typename MetricType, typename MatType>
void NCA<MetricType, MatType>::LearnDistance(MatType& output_matrix)
{
  output_matrix = arma::eye<MatType>(dataset_.n_rows, dataset_.n_rows);

  SoftmaxErrorFunction<MetricType> error_func(dataset_, labels_);

  // We will use the L-BFGS optimizer to optimize the stretching matrix.
  optimization::L_BFGS<SoftmaxErrorFunction<MetricType, MatType> >
      lbfgs(error_func, 10);

  lbfgs.Optimize(0, output_matrix);
}

}; // namespace nca
}; // namespace mlpack

#endif
