/**
 * @file tests/custom_layer.hpp
 * @author Projyal Dev
 *
 * A simple custom layer mimicking SigmoidLayer for testing if custom
 * layers work.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_CUSTOM_LAYER_HPP
#define MLPACK_TESTS_CUSTOM_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

namespace mlpack {
namespace ann {

/**
 * Standard Sigmoid layer.
 */
template <
    class ActivationFunction = LogisticFunction,
    typename MatType = arma::mat
>
using CustomLayer = BaseLayer<ActivationFunction, MatType>;

} // namespace ann
} // namespace mlpack

#endif // MLPACK_TESTS_CUSTOM_LAYER_HPP
