/**
 * @file serialization.hpp
 * @author Ryan Curtin
 *
 * Set up polymorphic serialization correctly for layer types.  If you need
 * custom serialization for a non-standard type, you will have to use the macros
 * in this file.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SERIALIZATION_HPP
#define MLPACK_METHODS_ANN_LAYER_SERIALIZATION_HPP

#define CEREAL_REGISTER_MLPACK_LAYERS(...) \
    CEREAL_REGISTER_TYPE(mlpack::ann::Layer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MultiLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::RecurrentLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AdaptiveMeanPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AdaptiveMaxPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AddType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AddMergeType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AlphaDropoutType<__VA_ARGS__>); \
    /* Base layers from base_layer.hpp. */ \
    CEREAL_REGISTER_TYPE(mlpack::ann::SigmoidType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::TanHType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SoftPlusType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::HardSigmoidType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SwishType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MishType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LiSHTType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::GELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ElliotType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ElishType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::GaussianType<__VA_ARGS__>); \
    /* (end of base_layer.hpp) */ \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConcatType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConcatenateType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConvolutionType< \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::FullConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        __VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::CELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::DropConnectType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::DropoutType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::IdentityType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LeakyReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::Linear3DType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearNoBiasType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LogSoftMaxType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LSTMType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MaxPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MeanPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::NoisyLinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::PaddingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::RBFType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SoftmaxType<__VA_ARGS__>); \

CEREAL_REGISTER_MLPACK_LAYERS(arma::mat);

#endif
