/***
 * @file mvu_impl.h
 * @author Ryan Curtin
 *
 * Implementation of the MVU class and its auxiliary objective function class.
 */

#ifndef __MLPACK_MVU_IMPL_H
#define __MLPACK_MVU_IMPL_H

#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

namespace mlpack {
namespace mvu {

template<typename LagrangianFunction>
MVU<LagrangianFunction>::MVU(arma::mat& data_in) :
    data_(data_in),
    f_(data_) {
  // Nothing to do.
}

template<typename LagrangianFunction>
bool MVU<LagrangianFunction>::Unfold(arma::mat& output_coordinates) {
  // Set up Augmented Lagrangian method.
  // Memory choice is arbitrary; this needs to be configurable.
  mlpack::optimization::AugLagrangian<LagrangianFunction> aug(f_, 20);

  output_coordinates = f_.GetInitialPoint();
  aug.Optimize(0, output_coordinates);

  return true;
}

}; // namespace mvu
}; // namespace mlpack

#endif
