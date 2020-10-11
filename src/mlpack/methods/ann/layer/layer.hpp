/**
 * @file methods/ann/layer/layer.hpp
 * @author Marcus Edel
 *
 * This includes various layers to construct a model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_HPP

#include "add.hpp"
#include "adaptive_max_pooling.hpp"
#include "adaptive_mean_pooling.hpp"
#include "add_merge.hpp"
#include "alpha_dropout.hpp"
#include "atrous_convolution.hpp"
#include "base_layer.hpp"
#include "batch_norm.hpp"
#include "bilinear_interpolation.hpp"
#include "c_relu.hpp"
#include "celu.hpp"
#include "concat_performance.hpp"
#include "concat.hpp"
#include "concatenate.hpp"
#include "constant.hpp"
#include "convolution.hpp"
#include "dropconnect.hpp"
#include "dropout.hpp"
#include "elu.hpp"
#include "fast_lstm.hpp"
#include "flexible_relu.hpp"
#include "glimpse.hpp"
#include "gru.hpp"
#include "hard_tanh.hpp"
#include "hardshrink.hpp"
#include "highway.hpp"
#include "join.hpp"
#include "layer_norm.hpp"
#include "layer_types.hpp"
#include "leaky_relu.hpp"
#include "linear.hpp"
#include "linear_no_bias.hpp"
#include "linear3d.hpp"
#include "log_softmax.hpp"
#include "lookup.hpp"
#include "lstm.hpp"
#include "max_pooling.hpp"
#include "mean_pooling.hpp"
#include "minibatch_discrimination.hpp"
#include "multihead_attention.hpp"
#include "multiply_constant.hpp"
#include "multiply_merge.hpp"
#include "noisylinear.hpp"
#include "padding.hpp"
#include "parametric_relu.hpp"
#include "positional_encoding.hpp"
#include "recurrent_attention.hpp"
#include "recurrent.hpp"
#include "reinforce_normal.hpp"
#include "reparametrization.hpp"
#include "select.hpp"
#include "sequential.hpp"
#include "softshrink.hpp"
#include "softmax.hpp"
#include "softmin.hpp"
#include "spatial_dropout.hpp"
#include "subview.hpp"
#include "transposed_convolution.hpp"
#include "virtual_batch_norm.hpp"
#include "vr_class_reward.hpp"
#include "weight_norm.hpp"

#endif
