
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_gan_gan.hpp:

Program Listing for File gan.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_gan_gan.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/gan/gan.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_GAN_GAN_HPP
   #define MLPACK_METHODS_ANN_GAN_GAN_HPP
   
   #include <mlpack/core.hpp>
   
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/gan/gan_policies.hpp>
   #include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>
   #include <mlpack/methods/ann/visitor/reset_visitor.hpp>
   #include <mlpack/methods/ann/visitor/weight_size_visitor.hpp>
   #include <mlpack/methods/ann/visitor/weight_set_visitor.hpp>
   #include "metrics/inception_score.hpp"
   
   
   namespace mlpack {
   namespace ann  {
   
   template<
     typename Model,
     typename InitializationRuleType,
     typename Noise,
     typename PolicyType = StandardGAN
   >
   class GAN
   {
    public:
     GAN(Model generator,
         Model discriminator,
         InitializationRuleType& initializeRule,
         Noise& noiseFunction,
         const size_t noiseDim,
         const size_t batchSize,
         const size_t generatorUpdateStep,
         const size_t preTrainSize,
         const double multiplier,
         const double clippingParameter = 0.01,
         const double lambda = 10.0);
   
     GAN(const GAN&);
   
     GAN(GAN&&);
   
     void ResetData(arma::mat trainData);
   
     // Reset function.
     void Reset();
   
     template<typename OptimizerType, typename... CallbackTypes>
     double Train(arma::mat trainData,
                  OptimizerType& Optimizer,
                  CallbackTypes&&... callbacks);
   
     template<typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                             std::is_same<Policy, DCGAN>::value, double>::type
     Evaluate(const arma::mat& parameters,
              const size_t i,
              const size_t batchSize);
   
     template<typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, WGAN>::value,
                             double>::type
     Evaluate(const arma::mat& parameters,
              const size_t i,
              const size_t batchSize);
   
     template<typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                             double>::type
     Evaluate(const arma::mat& parameters,
              const size_t i,
              const size_t batchSize);
   
     template<typename GradType, typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                             std::is_same<Policy, DCGAN>::value, double>::type
     EvaluateWithGradient(const arma::mat& parameters,
                          const size_t i,
                          GradType& gradient,
                          const size_t batchSize);
   
     template<typename GradType, typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, WGAN>::value,
                             double>::type
     EvaluateWithGradient(const arma::mat& parameters,
                          const size_t i,
                          GradType& gradient,
                          const size_t batchSize);
   
     template<typename GradType, typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                             double>::type
     EvaluateWithGradient(const arma::mat& parameters,
                          const size_t i,
                          GradType& gradient,
                          const size_t batchSize);
   
     template<typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, StandardGAN>::value ||
                             std::is_same<Policy, DCGAN>::value, void>::type
     Gradient(const arma::mat& parameters,
              const size_t i,
              arma::mat& gradient,
              const size_t batchSize);
   
     template<typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, WGAN>::value, void>::type
     Gradient(const arma::mat& parameters,
              const size_t i,
              arma::mat& gradient,
              const size_t batchSize);
   
     template<typename Policy = PolicyType>
     typename std::enable_if<std::is_same<Policy, WGANGP>::value,
                             void>::type
     Gradient(const arma::mat& parameters,
              const size_t i,
              arma::mat& gradient,
              const size_t batchSize);
   
     void Shuffle();
   
     void Forward(const arma::mat& input);
   
     void Predict(arma::mat input, arma::mat& output);
   
     const arma::mat& Parameters() const { return parameter; }
     arma::mat& Parameters() { return parameter; }
   
     const Model& Generator() const { return generator; }
     Model& Generator() { return generator; }
     const Model& Discriminator() const { return discriminator; }
     Model& Discriminator() { return discriminator; }
   
     size_t NumFunctions() const { return numFunctions; }
   
     const arma::mat& Responses() const { return responses; }
     arma::mat& Responses() { return responses; }
   
     const arma::mat& Predictors() const { return predictors; }
     arma::mat& Predictors() { return predictors; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     void ResetDeterministic();
   
     arma::mat predictors;
     arma::mat parameter;
     Model generator;
     Model discriminator;
     InitializationRuleType initializeRule;
     Noise noiseFunction;
     size_t noiseDim;
     size_t numFunctions;
     size_t batchSize;
     size_t currentBatch;
     size_t generatorUpdateStep;
     size_t preTrainSize;
     double multiplier;
     double clippingParameter;
     double lambda;
     bool reset;
     DeltaVisitor deltaVisitor;
     arma::mat responses;
     arma::mat currentInput;
     arma::mat currentTarget;
     OutputParameterVisitor outputParameterVisitor;
     WeightSizeVisitor weightSizeVisitor;
     ResetVisitor resetVisitor;
     arma::mat gradient;
     arma::mat gradientDiscriminator;
     arma::mat noiseGradientDiscriminator;
     arma::mat normGradientDiscriminator;
     arma::mat noise;
     arma::mat gradientGenerator;
     bool deterministic;
     size_t genWeights;
     size_t discWeights;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "gan_impl.hpp"
   #include "wgan_impl.hpp"
   #include "wgangp_impl.hpp"
   
   
   #endif
