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

/*
#define CEREAL_REGISTER_MLPACK_LAYERS(...) \
    CEREAL_REGISTER_TYPE(mlpack::ann::AdaptiveMaxPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AdaptiveMeanPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AddType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AddMerge<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AlphaDropout<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AtrousConvolution< \
        / TODO: change ordering of template types? / \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::FullConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        __VA_ARGS__>); \
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
    CEREAL_REGISTER_TYPE(mlpack::ann::BatchNorm<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::BilinearInterpolationType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::CELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConcatenateType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConcatType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConcatPerformance< \
        mlpack::ann::NegativeLogLikelihood<>, __VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConstantType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConvolutionType< \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::FullConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        __VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::CReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::DropConnectType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::DropoutType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ELUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::FastLSTMType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::FlexibleReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::GlimpseType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::GRU<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::HardShrinkType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::HardTanHType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::HighwayType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::JoinType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::Layer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LayerNormType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LeakyReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::Linear3DType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearNoBiasType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LogSoftMaxType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LSTM<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MaxPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MeanPoolingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MiniBatchDiscrimination<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MultiheadAttentionType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MultiplyConstantType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MultiplyMergeType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::NoisyLinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::PaddingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::PReLUType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::PositionalEncodingType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::RBF<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::RecurrentAttention<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::Recurrent<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ReinforceNormalType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ReparametrizationType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SelectType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SequentialType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SoftmaxType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SoftminType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SoftShrinkType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SpatialDropoutType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::SubviewType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::TransposedConvolutionType< \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        mlpack::ann::NaiveConvolution<mlpack::ann::ValidConvolution>, \
        __VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::VirtualBatchNormType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::WeightNormType<__VA_ARGS__>);
*/

#define CEREAL_REGISTER_MLPACK_LAYERS(...) \
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
    CEREAL_REGISTER_TYPE(mlpack::ann::DropConnectType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::DropoutType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::Layer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::MultiLayer<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LinearNoBiasType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::LogSoftMaxType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ReparametrizationType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::Linear3DType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::NoisyLinearType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::ConcatenateType<__VA_ARGS__>); \
    CEREAL_REGISTER_TYPE(mlpack::ann::AddType<__VA_ARGS__>); \

// TODO: continue...

CEREAL_REGISTER_MLPACK_LAYERS(arma::mat, arma::mat);

// TODO: I think this below is not needed.
/**
 * Register an mlpack layer with the given INPUT_TYPE and OUTPUT_TYPE.
 * Unfortunately, due to limitations of the C++ preprocessor, INPUT_TYPE and
 * OUTPUT_TYPE *cannot* have commas in them.  In general, INPUT_TYPE and
 * OUTPUT_TYPE will be Armadillo matrix types holding different element types;
 * e.g., `arma::mat`, `arma::Mat<float>`, `arma::sp_mat`, and so forth.
 *
 * This will register the type LAYER<INPUT_TYPE, OUTPUT_TYPE> for serialization.
 * If you use a LAYER<INPUT_TYPE, OUTPUT_TYPE> in your program and plan to use
 * serialization, this macro *must* be present in your program.
 *
 * By default, mlpack automatically registers all internal layer types with
 * `arma::mat` as `INPUT_TYPE` and `OUTPUT_TYPE`.
 *
 * This macro must be called in the global namespace.
 *
 * @param LAYER Type of layer to register for polymorphic serialization.  Be
 *     sure to use the fully-qualified name.
 * @param INPUT_TYPE Type of input to layer.
 * @param OUTPUT_TYPE Type of output from layer.
 */
//#define CEREAL_REGISTER_MLPACK_LAYER(LAYER, INPUT_TYPE, OUTPUT_TYPE) 
//    /* This is a copy of CEREAL_REGISTER_TYPE, adapted for template */ 
//    /* parameters. */ 
//    namespace cereal { 
//    namespace detail { 
//    template<> 
//    struct binding_name<LAYER<INPUT_TYPE, OUTPUT_TYPE>> 
//    { 
//      CEREAL_STATIC_CONSTEXPR char const* name() 
//      { 
//        return #LAYER "<" #INPUT_TYPE "," #OUTPUT_TYPE ">" 
//      } 
//    }; 
//    /* This is a copy of CEREAL_BIND_TO_ARCHIVES, adapted for template */ 
//    /* parameters. */ 
//    template<> 
//    struct init_binding<LAYER<INPUT_TYPE, OUTPUT_TYPE>> {
//        
//    } } /* end namespaces */ 
//    

#endif
