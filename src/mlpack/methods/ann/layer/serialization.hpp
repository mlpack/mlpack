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
    CEREAL_REGISTER_TYPE(mlpack::ann::AddType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AlphaDropout<__VA_ARGS__>); \
    /* Base layers from base_layer.hpp. */ \
    CEREAL_REGISTER_TYPE(mlpack::ann::SigmoidLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::IdentityLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ReLULayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::TanHLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SoftPlusLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::HardSigmoidLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SwishFunctionLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MishFunctionLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LiSHTFunctionLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::GELUFunctionLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ElliotFunctionLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ElishFunctionLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::GaussianFunctionLayer<__VA_ARGS__>); \
    /* (end of base_layer.hpp) */ \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConcatenateType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConvolutionType< \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::FullConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        __VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::DropConnectType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::DropoutType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LeakyReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::Linear3DType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearNoBiasType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LogSoftMaxType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MaxPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::NoisyLinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::PaddingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::RBF<__VA_ARGS__>); \

CEREAL_REGISTER_MLPACK_LAYERS(arma::mat, arma::mat);

#endif
