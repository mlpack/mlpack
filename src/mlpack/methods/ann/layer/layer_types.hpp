/**
 * @file methods/ann/layer/layer_types.hpp
 * @author Marcus Edel
 *
 * This provides a list of all modules that can be used to construct a model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_TYPES_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_TYPES_HPP

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/multi_layer.hpp>
#include <mlpack/methods/ann/layer/recurrent_layer.hpp>

// Include each layer.
#include <mlpack/methods/ann/layer/adaptive_max_pooling.hpp>
#include <mlpack/methods/ann/layer/adaptive_mean_pooling.hpp>
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/add_merge.hpp>
#include <mlpack/methods/ann/layer/alpha_dropout.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/celu.hpp>
#include <mlpack/methods/ann/layer/c_relu.hpp>
#include <mlpack/methods/ann/layer/concat.hpp>
#include <mlpack/methods/ann/layer/concatenate.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/dropconnect.hpp>
#include <mlpack/methods/ann/layer/dropout.hpp>
#include <mlpack/methods/ann/layer/elu.hpp>
#include <mlpack/methods/ann/layer/flexible_relu.hpp>
#include <mlpack/methods/ann/layer/grouped_convolution.hpp>
#include <mlpack/methods/ann/layer/hard_tanh.hpp>
#include <mlpack/methods/ann/layer/identity.hpp>
#include <mlpack/methods/ann/layer/layer_norm.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/linear_no_bias.hpp>
#include <mlpack/methods/ann/layer/linear_recurrent.hpp>
#include <mlpack/methods/ann/layer/linear3d.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>
#include <mlpack/methods/ann/layer/lstm.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/mean_pooling.hpp>
#include <mlpack/methods/ann/layer/multihead_attention.hpp>
#include <mlpack/methods/ann/layer/nearest_interpolation.hpp>
#include <mlpack/methods/ann/layer/noisylinear.hpp>
#include <mlpack/methods/ann/layer/padding.hpp>
#include <mlpack/methods/ann/layer/parametric_relu.hpp>
#include <mlpack/methods/ann/layer/radial_basis_function.hpp>
#include <mlpack/methods/ann/layer/relu6.hpp>
#include <mlpack/methods/ann/layer/repeat.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/methods/ann/layer/softmin.hpp>
#include <mlpack/methods/ann/layer/ftswish.hpp>

// Convolution modes.
#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>

// Regularizers.
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

// Loss function modules.
#include <mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp>

// Include definitions for polymorphic serialization.  Note that this can cause
// significant compilation overhead, so we only do it if
// MLPACK_ENABLE_ANN_SERIALIZATION is enabled.
#ifdef MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack/methods/ann/layer/serialization.hpp>
#endif

#endif
