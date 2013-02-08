/**
 * @file nca_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated NCA class.
 *
 * This file is part of MLPACK 1.0.4.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_NCA_NCA_IMPL_HPP
#define __MLPACK_METHODS_NCA_NCA_IMPL_HPP

// In case it was not already included.
#include "nca.hpp"

namespace mlpack {
namespace nca {

// Just set the internal matrix reference.
template<typename MetricType, template<typename> class OptimizerType>
NCA<MetricType, OptimizerType>::NCA(const arma::mat& dataset,
                                    const arma::uvec& labels,
                                    MetricType metric) :
    dataset(dataset),
    labels(labels),
    metric(metric),
    errorFunction(dataset, labels, metric),
    optimizer(OptimizerType<SoftmaxErrorFunction<MetricType> >(errorFunction))
{ /* Nothing to do. */ }

template<typename MetricType, template<typename> class OptimizerType>
void NCA<MetricType, OptimizerType>::LearnDistance(arma::mat& outputMatrix)
{
  // See if we were passed an initialized matrix.
  if ((outputMatrix.n_rows != dataset.n_rows) ||
      (outputMatrix.n_cols != dataset.n_rows))
    outputMatrix.eye(dataset.n_rows, dataset.n_rows);

  Timer::Start("nca_sgd_optimization");

  optimizer.Optimize(outputMatrix);

  Timer::Stop("nca_sgd_optimization");
}

}; // namespace nca
}; // namespace mlpack

#endif
