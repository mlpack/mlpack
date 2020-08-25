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

#include <boost/variant.hpp>

// Layer modules.
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/alpha_dropout.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/bilinear_interpolation.hpp>
#include <mlpack/methods/ann/layer/constant.hpp>
#include <mlpack/methods/ann/layer/concatenate.hpp>
#include <mlpack/methods/ann/layer/dropout.hpp>
#include <mlpack/methods/ann/layer/elu.hpp>
#include <mlpack/methods/ann/layer/hard_tanh.hpp>
#include <mlpack/methods/ann/layer/join.hpp>
#include <mlpack/methods/ann/layer/layer_norm.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/c_relu.hpp>
#include <mlpack/methods/ann/layer/flexible_relu.hpp>
#include <mlpack/methods/ann/layer/linear_no_bias.hpp>
#include <mlpack/methods/ann/layer/linear3d.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>
#include <mlpack/methods/ann/layer/lookup.hpp>
#include <mlpack/methods/ann/layer/multihead_attention.hpp>
#include <mlpack/methods/ann/layer/multiply_constant.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/mean_pooling.hpp>
#include <mlpack/methods/ann/layer/noisylinear.hpp>
#include <mlpack/methods/ann/layer/adaptive_max_pooling.hpp>
#include <mlpack/methods/ann/layer/adaptive_mean_pooling.hpp>
#include <mlpack/methods/ann/layer/parametric_relu.hpp>
#include <mlpack/methods/ann/layer/positional_encoding.hpp>
#include <mlpack/methods/ann/layer/reinforce_normal.hpp>
#include <mlpack/methods/ann/layer/reparametrization.hpp>
#include <mlpack/methods/ann/layer/select.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/methods/ann/layer/spatial_dropout.hpp>
#include <mlpack/methods/ann/layer/subview.hpp>
#include <mlpack/methods/ann/layer/virtual_batch_norm.hpp>
#include <mlpack/methods/ann/layer/hardshrink.hpp>
#include <mlpack/methods/ann/layer/celu.hpp>
#include <mlpack/methods/ann/layer/softshrink.hpp>
#include <mlpack/methods/ann/layer/radial_basis_function.hpp>

// Convolution modules.
#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>

// Regularizers.
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

// Loss function modules.
#include <mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp>

namespace mlpack {
namespace ann {

template<typename InputDataType, typename OutputDataType> class BatchNorm;
template<typename InputDataType, typename OutputDataType> class DropConnect;
template<typename InputDataType, typename OutputDataType> class Glimpse;
template<typename InputDataType, typename OutputDataType> class LayerNorm;
template<typename InputDataType, typename OutputDataType> class LSTM;
template<typename InputDataType, typename OutputDataType> class GRU;
template<typename InputDataType, typename OutputDataType> class FastLSTM;
template<typename InputDataType, typename OutputDataType> class VRClassReward;
template<typename InputDataType, typename OutputDataType> class Concatenate;
template<typename InputDataType, typename OutputDataType> class Padding;

template<typename InputDataType,
         typename OutputDataType,
         typename RegularizerType>
class Linear;

template<typename InputDataType,
         typename OutputDataType,
         typename Activation>
class RBF;

template<typename InputDataType,
         typename OutputDataType,
         typename RegularizerType>
class LinearNoBias;

template<typename InputDataType,
         typename OutputDataType>
class NoisyLinear;

template<typename InputDataType,
         typename OutputDataType,
         typename RegularizerType>
class Linear3D;

template<typename InputDataType,
         typename OutputDataType
>
class VirtualBatchNorm;

template<typename InputDataType,
         typename OutputDataType
>
class MiniBatchDiscrimination;

template <typename InputDataType,
          typename OutputDataType,
          typename RegularizerType>
class MultiheadAttention;

template<typename InputDataType,
         typename OutputDataType
>
class Reparametrization;

template<typename InputDataType,
         typename OutputDataType,
         typename... CustomLayers
>
class AddMerge;

template<typename InputDataType,
         typename OutputDataType,
         bool residual,
         typename... CustomLayers
>
class Sequential;

template<typename InputDataType,
         typename OutputDataType,
         typename... CustomLayers
>
class Highway;

template<typename InputDataType,
         typename OutputDataType,
         typename... CustomLayers
>
class Recurrent;

template<typename InputDataType,
         typename OutputDataType,
         typename... CustomLayers
>
class Concat;

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
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
class TransposedConvolution;

template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename InputDataType,
    typename OutputDataType
>
class AtrousConvolution;

template<
    typename InputDataType,
    typename OutputDataType
>
class RecurrentAttention;

template<typename InputDataType,
         typename OutputDataType,
         typename... CustomLayers
>
class MultiplyMerge;

template <typename InputDataType,
          typename OutputDataType,
          typename... CustomLayers
>
class WeightNorm;

template <typename InputDataType,
          typename OutputDataType
>
class AdaptiveMaxPooling;

template <typename InputDataType,
          typename OutputDataType
>
class AdaptiveMeanPooling;

using MoreTypes = boost::variant<
        Linear3D<arma::mat, arma::mat, NoRegularizer>*,
        Glimpse<arma::mat, arma::mat>*,
        Highway<arma::mat, arma::mat>*,
        MultiheadAttention<arma::mat, arma::mat, NoRegularizer>*,
        Recurrent<arma::mat, arma::mat>*,
        RecurrentAttention<arma::mat, arma::mat>*,
        ReinforceNormal<arma::mat, arma::mat>*,
        Reparametrization<arma::mat, arma::mat>*,
        Select<arma::mat, arma::mat>*,
        Sequential<arma::mat, arma::mat, false>*,
        Sequential<arma::mat, arma::mat, true>*,
        Subview<arma::mat, arma::mat>*,
        VRClassReward<arma::mat, arma::mat>*,
        VirtualBatchNorm<arma::mat, arma::mat>*,
        RBF<arma::mat, arma::mat, GaussianFunction>*,
        BaseLayer<GaussianFunction, arma::mat, arma::mat>*,
        PositionalEncoding<arma::mat, arma::mat>*
>;

template <typename... CustomLayers>
using LayerTypes = boost::variant<
    AdaptiveMaxPooling<arma::mat, arma::mat>*,
    AdaptiveMeanPooling<arma::mat, arma::mat>*,
    Add<arma::mat, arma::mat>*,
    AddMerge<arma::mat, arma::mat>*,
    AlphaDropout<arma::mat, arma::mat>*,
    AtrousConvolution<NaiveConvolution<ValidConvolution>,
                      NaiveConvolution<FullConvolution>,
                      NaiveConvolution<ValidConvolution>,
                      arma::mat, arma::mat>*,
    BaseLayer<LogisticFunction, arma::mat, arma::mat>*,
    BaseLayer<IdentityFunction, arma::mat, arma::mat>*,
    BaseLayer<TanhFunction, arma::mat, arma::mat>*,
    BaseLayer<SoftplusFunction, arma::mat, arma::mat>*,
    BaseLayer<RectifierFunction, arma::mat, arma::mat>*,
    BatchNorm<arma::mat, arma::mat>*,
    BilinearInterpolation<arma::mat, arma::mat>*,
    CELU<arma::mat, arma::mat>*,
    Concat<arma::mat, arma::mat>*,
    Concatenate<arma::mat, arma::mat>*,
    ConcatPerformance<NegativeLogLikelihood<arma::mat, arma::mat>,
                      arma::mat, arma::mat>*,
    Constant<arma::mat, arma::mat>*,
    Convolution<NaiveConvolution<ValidConvolution>,
                NaiveConvolution<FullConvolution>,
                NaiveConvolution<ValidConvolution>, arma::mat, arma::mat>*,
    CReLU<arma::mat, arma::mat>*,
    DropConnect<arma::mat, arma::mat>*,
    Dropout<arma::mat, arma::mat>*,
    ELU<arma::mat, arma::mat>*,
    FastLSTM<arma::mat, arma::mat>*,
    FlexibleReLU<arma::mat, arma::mat>*,
    GRU<arma::mat, arma::mat>*,
    HardTanH<arma::mat, arma::mat>*,
    Join<arma::mat, arma::mat>*,
    LayerNorm<arma::mat, arma::mat>*,
    LeakyReLU<arma::mat, arma::mat>*,
    Linear<arma::mat, arma::mat, NoRegularizer>*,
    LinearNoBias<arma::mat, arma::mat, NoRegularizer>*,
    LogSoftMax<arma::mat, arma::mat>*,
    Lookup<arma::mat, arma::mat>*,
    LSTM<arma::mat, arma::mat>*,
    MaxPooling<arma::mat, arma::mat>*,
    MeanPooling<arma::mat, arma::mat>*,
    MiniBatchDiscrimination<arma::mat, arma::mat>*,
    MultiplyConstant<arma::mat, arma::mat>*,
    MultiplyMerge<arma::mat, arma::mat>*,
    NegativeLogLikelihood<arma::mat, arma::mat>*,
    NoisyLinear<arma::mat, arma::mat>*,
    Padding<arma::mat, arma::mat>*,
    PReLU<arma::mat, arma::mat>*,
    Softmax<arma::mat, arma::mat>*,
    SpatialDropout<arma::mat, arma::mat>*,
    TransposedConvolution<NaiveConvolution<ValidConvolution>,
            NaiveConvolution<ValidConvolution>,
            NaiveConvolution<ValidConvolution>, arma::mat, arma::mat>*,
    WeightNorm<arma::mat, arma::mat>*,
    MoreTypes,
    CustomLayers*...
>;

} // namespace ann
} // namespace mlpack

#endif
