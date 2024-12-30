/**
 * @file serialization.hpp
 * @author Ryan Curtin
 *
 * Set up polymorphic serialization correctly for layer types.  If you need
 * custom serialization for a non-standard type, you will have to use the macros
 * in this file.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SERIALIZATION_HPP
#define MLPACK_METHODS_ANN_LAYER_SERIALIZATION_HPP

#define CEREAL_REGISTER_MLPACK_LAYERS(...) \
    CEREAL_REGISTER_TYPE(mlpack::Layer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MultiLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::RecurrentLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AdaptiveMeanPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AdaptiveMaxPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AddType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AddMergeType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AlphaDropoutType<__VA_ARGS__>); \
    /* Base layers from base_layer.hpp. */ \
    CEREAL_REGISTER_TYPE(mlpack::SigmoidType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::TanHType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::SoftPlusType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::HardSigmoidType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::SwishType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MishType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LiSHTType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::GELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ElliotType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ElishType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::GaussianType<__VA_ARGS__>); \
    /* (end of base_layer.hpp) */ \
    CEREAL_REGISTER_TYPE(mlpack::BatchNormType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ConcatType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ConcatenateType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ConvolutionType< \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>, \
        mlpack::NaiveConvolution<mlpack::FullConvolution>, \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>, \
        __VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::CELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::CReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::DropConnectType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::DropoutType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::FlexibleReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::GroupedConvolutionType< \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>, \
        mlpack::NaiveConvolution<mlpack::FullConvolution>, \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>, \
        __VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::IdentityType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LeakyReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LayerNormType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Linear3DType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LinearNoBiasType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LinearRecurrentType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LogSoftMaxType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LSTMType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MaxPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MeanPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MultiheadAttentionType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::NoisyLinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::PaddingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::PReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::RBFType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ReLU6Type<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::RepeatType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::SoftmaxType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::SoftminType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::HardTanHType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::FTSwishType<__VA_ARGS__>); \

CEREAL_REGISTER_MLPACK_LAYERS(arma::mat);

#endif
