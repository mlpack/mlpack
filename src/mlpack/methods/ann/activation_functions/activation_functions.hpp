/**
 * @file methods/ann/activation_functions/activation_functions.hpp"
 * @author Ryan Curtin
 *
 * Convenience include for all activation functions implemented for mlpack's
 * neural network toolkit.
 *
 * An activation function should define methods to evaluate the function
 * and its derivative.
 *
 * For the forward pass, a class should define
 * static double Fn(double x) -- evaluate y = F(x) at a single point
 * and
 * static void Fn(const InputVecType& x, OutputVecType& y) -- evaluate y = F(x)
 * for a vector
 *
 * For the backward pass, a class should define the derivative function.  For
 * efficiency of implementation, it will be provided both x (the inputs) and
 * y (the result of F(x)).  The following should be defined
 * static double Deriv(double x, double y) -- evaluate dF(x)/dx for one value
 * of x given both x and y=F(x)
 * static void Deriv(const InputVecType& x, const OutputVecType& y,
 *                   DerivVecType& dy) -- evaluate dF(x)/dx for a vector x
 *                                        and a vector y=F(x)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ACTIVATION_FUNCTIONS_HPP
#define MLPACK_METHODS_ANN_ACTIVATION_FUNCTIONS_ACTIVATION_FUNCTIONS_HPP

#include "elish_function.hpp"
#include "elliot_function.hpp"
#include "gaussian_function.hpp"
#include "gelu_function.hpp"
#include "hard_sigmoid_function.hpp"
#include "hard_swish_function.hpp"
#include "identity_function.hpp"
#include "inverse_quadratic_function.hpp"
#include "lisht_function.hpp"
#include "logistic_function.hpp"
#include "mish_function.hpp"
#include "multi_quadratic_function.hpp"
#include "poisson1_function.hpp"
#include "quadratic_function.hpp"
#include "rectifier_function.hpp"
#include "silu_function.hpp"
#include "softplus_function.hpp"
#include "softsign_function.hpp"
#include "spline_function.hpp"
#include "swish_function.hpp"
#include "tanh_exponential_function.hpp"
#include "tanh_function.hpp"
#include "hyper_sinh_function.hpp"
#include "bipolar_sigmoid_function.hpp"

#endif
