#ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_HPP
#define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include "softmax_regression_function.hpp"

namespace mlpack {
namespace neural_network {

template<typename... LayerClasses>
class NeuralNetwork
{
 private:
  const size_t Layers = sizeof...(Types);
}

}
}
