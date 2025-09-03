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
    CEREAL_REGISTER_TYPE(mlpack::AdaptiveMeanPooling<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AdaptiveMaxPooling<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Add<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AddMerge<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::AlphaDropout<__VA_ARGS__>); \
    /* Base layers from base_layer.hpp. */ \
    CEREAL_REGISTER_TYPE(mlpack::Sigmoid<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ReLU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::TanH<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::SoftPlus<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::HardSigmoid<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Swish<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Mish<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LiSHT<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::GELU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::GELUExact<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Elliot<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Elish<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Gaussian<__VA_ARGS__>); \
    /* (end of base_layer.hpp) */ \
    CEREAL_REGISTER_TYPE(mlpack::BatchNorm<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::CELU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Concat<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Concatenate<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Convolution<__VA_ARGS__, \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>, \
        mlpack::NaiveConvolution<mlpack::FullConvolution>, \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>>); \
    CEREAL_REGISTER_TYPE(mlpack::CReLU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::DropConnect<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Dropout<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ELU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::FlexibleReLU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::GroupedConvolution<__VA_ARGS__, \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>, \
        mlpack::NaiveConvolution<mlpack::FullConvolution>, \
        mlpack::NaiveConvolution<mlpack::ValidConvolution>>); \
    CEREAL_REGISTER_TYPE(mlpack::GRU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Identity<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LeakyReLU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LayerNorm<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Linear3D<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Linear<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LinearNoBias<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LinearRecurrent<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LogSoftMax<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::LSTM<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MaxPooling<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MeanPooling<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::MultiheadAttention<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::NoisyLinear<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Padding<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::PReLU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::RBF<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ReLU6<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Repeat<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Softmax<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::Softmin<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::HardTanH<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::FTSwish<__VA_ARGS__>); \

CEREAL_REGISTER_MLPACK_LAYERS(arma::mat);

#endif
