/**
 * @file nca_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated NCA class.
 */
#ifndef __MLPACK_METHODS_NCA_NCA_IMPL_HPP
#define __MLPACK_METHODS_NCA_NCA_IMPL_HPP

// In case it was not already included.
#include "nca.hpp"

#include <mlpack/core/optimizers/sgd/sgd.hpp>

#include "nca_softmax_error_function.hpp"

namespace mlpack {
namespace nca {

// Just set the internal matrix reference.
template<typename MetricType>
NCA<MetricType>::NCA(const arma::mat& dataset,
                     const arma::uvec& labels,
                     const double stepSize,
                     const size_t maxIterations,
                     const double tolerance,
                     const bool shuffle,
                     MetricType metric) :
    dataset(dataset),
    labels(labels),
    metric(metric),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

template<typename MetricType>
void NCA<MetricType>::LearnDistance(arma::mat& outputMatrix)
{
  // See if we were passed an initialized matrix.
  if ((outputMatrix.n_rows != dataset.n_rows) ||
      (outputMatrix.n_cols != dataset.n_rows))
    outputMatrix.eye(dataset.n_rows, dataset.n_rows);

  SoftmaxErrorFunction<MetricType> errorFunc(dataset, labels, metric);

  // We will use stochastic gradient descent to optimize the NCA error function.
  optimization::SGD<SoftmaxErrorFunction<MetricType> > sgd(errorFunc, stepSize,
      maxIterations, tolerance, shuffle);

  Timer::Start("nca_sgd_optimization");

  sgd.Optimize(outputMatrix);

  Timer::Stop("nca_sgd_optimization");
}

}; // namespace nca
}; // namespace mlpack

#endif
