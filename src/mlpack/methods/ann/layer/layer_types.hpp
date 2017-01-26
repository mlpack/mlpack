/**
 * @file layer_types.hpp
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

#include <boost/variant.hpp>

// Layer modules.
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/constant.hpp>
#include <mlpack/methods/ann/layer/dropout.hpp>
#include <mlpack/methods/ann/layer/hard_tanh.hpp>
#include <mlpack/methods/ann/layer/join.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>
#include <mlpack/methods/ann/layer/lookup.hpp>
#include <mlpack/methods/ann/layer/mean_squared_error.hpp>
#include <mlpack/methods/ann/layer/multiply_constant.hpp>
#include <mlpack/methods/ann/layer/negative_log_likelihood.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/mean_pooling.hpp>
#include <mlpack/methods/ann/layer/reinforce_normal.hpp>
#include <mlpack/methods/ann/layer/select.hpp>

// Convolution modules.
#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>

namespace mlpack {
namespace ann {

template<typename InputDataType, typename OutputDataType> class AddMerge;
template<typename InputDataType, typename OutputDataType> class Concat;
template<typename InputDataType, typename OutputDataType> class DropConnect;
template<typename InputDataType, typename OutputDataType> class Glimpse;
template<typename InputDataType, typename OutputDataType> class Linear;
template<typename InputDataType, typename OutputDataType> class LinearNoBias;
template<typename InputDataType, typename OutputDataType> class LSTM;
template<typename InputDataType, typename OutputDataType> class Recurrent;
template<typename InputDataType, typename OutputDataType> class Sequential;
template<typename InputDataType, typename OutputDataType> class VRClassReward;

template<
    typename OutputLayerType,
    typename InputDataType,
    typename OutputDataType
>
class ConcatPerformance;

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
class Convolution;

template<
    typename InputDataType,
    typename OutputDataType
>
class RecurrentAttention;

using LayerTypes = boost::variant<
    Add<arma::mat, arma::mat>*,
    AddMerge<arma::mat, arma::mat>*,
    BaseLayer<LogisticFunction, arma::mat, arma::mat>*,
    BaseLayer<IdentityFunction, arma::mat, arma::mat>*,
    BaseLayer<TanhFunction, arma::mat, arma::mat>*,
    BaseLayer<RectifierFunction, arma::mat, arma::mat>*,
    Concat<arma::mat, arma::mat>*,
    ConcatPerformance<NegativeLogLikelihood<arma::mat, arma::mat>,
                      arma::mat, arma::mat>*,
    Constant<arma::mat, arma::mat>*,
    Convolution<NaiveConvolution<ValidConvolution>,
                NaiveConvolution<FullConvolution>,
                NaiveConvolution<ValidConvolution>, arma::mat, arma::mat>*,
    DropConnect<arma::mat, arma::mat>*,
    Dropout<arma::mat, arma::mat>*,
    Glimpse<arma::mat, arma::mat>*,
    HardTanH<arma::mat, arma::mat>*,
    Join<arma::mat, arma::mat>*,
    LeakyReLU<arma::mat, arma::mat>*,
    Linear<arma::mat, arma::mat>*,
    LinearNoBias<arma::mat, arma::mat>*,
    LogSoftMax<arma::mat, arma::mat>*,
    Lookup<arma::mat, arma::mat>*,
    LSTM<arma::mat, arma::mat>*,
    MaxPooling<arma::mat, arma::mat>*,
    MeanPooling<arma::mat, arma::mat>*,
    MeanSquaredError<arma::mat, arma::mat>*,
    MultiplyConstant<arma::mat, arma::mat>*,
    NegativeLogLikelihood<arma::mat, arma::mat>*,
    Recurrent<arma::mat, arma::mat>*,
    RecurrentAttention<arma::mat, arma::mat>*,
    ReinforceNormal<arma::mat, arma::mat>*,
    Select<arma::mat, arma::mat>*,
    Sequential<arma::mat, arma::mat>*,
    VRClassReward<arma::mat, arma::mat>*
>;

} // namespace ann
} // namespace mlpack

#endif
