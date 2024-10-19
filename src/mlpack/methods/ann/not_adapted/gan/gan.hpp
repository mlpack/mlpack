/**
 * @file methods/ann/gan/gan.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
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

/**
 * The implementation of the standard GAN module. Generative Adversarial
 * Networks (GANs) are a class of artificial intelligence algorithms used
 * in unsupervised machine learning, implemented by a system of two neural
 * networks contesting with each other in a zero-sum game framework. This
 * technique can generate photographs that look at least superficially
 * authentic to human observers, having many realistic characteristics.
 * GANs have been used in Text-to-Image Synthesis, Medical Drug Discovery,
 * High Resolution Imagery Generation, Neural Machine Translation and so on.
 *
 * For more information, see the following paper:
 *
 * @code
 * @article{Goodfellow14,
 *   author    = {Ian J. Goodfellow, Jean Pouget-Abadi, Mehdi Mirza, Bing Xu,
 *                David Warde-Farley, Sherjil Ozair, Aaron Courville and
 *                Yoshua Bengio},
 *   title     = {Generative Adversarial Nets},
 *   year      = {2014},
 *   url       = {http://arxiv.org/abs/1406.2661},
 *   eprint    = {1406.2661},
 * }
 * @endcode
 *
 * @tparam Model The class type of Generator and Discriminator.
 * @tparam InitializationRuleType Type of Initializer.
 * @tparam Noise The noise function to use.
 * @tparam PolicyType The GAN variant to be used (GAN, DCGAN, WGAN or WGANGP).
 */
template<
  typename Model,
  typename InitializationRuleType,
  typename Noise,
  typename PolicyType = StandardGAN
>
class GAN
{
 public:
  /**
   * Constructor for GAN class.
   *
   * @param generator Generator network.
   * @param discriminator Discriminator network.
   * @param initializeRule Initialization rule to use for initializing
   *                       parameters.
   * @param noiseFunction Function to be used for generating noise.
   * @param noiseDim Dimension of noise vector to be created.
   * @param batchSize Batch size to be used for training.
   * @param generatorUpdateStep Number of steps to train Discriminator
   *                            before updating Generator.
   * @param preTrainSize Number of pre-training steps of Discriminator.
   * @param multiplier Ratio of learning rate of Discriminator to the Generator.
   * @param clippingParameter Weight range for enforcing Lipschitz constraint.
   * @param lambda Parameter for setting the gradient penalty.
   */
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

  //! Copy constructor.
  GAN(const GAN&);

  //! Move constructor.
  GAN(GAN&&);

  /**
   * Initialize the generator, discriminator and weights of the model for
   * training. This function won't actually trigger training process.
   *
   * @param trainData The data points of real distribution.
   */
  void ResetData(arma::mat trainData);

  // Reset function.
  void Reset();

  /**
   * Train function.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback functions.
   * @param trainData The data points of real distribution.
   * @param Optimizer Instantiated optimizer used to train the model. 
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackTypes>
  double Train(arma::mat trainData,
               OptimizerType& Optimizer,
               CallbackTypes&&... callbacks);

  /**
   * Evaluate function for the Standard GAN and DCGAN.
   * This function gives the performance of the Standard GAN or DCGAN on the
   * current input.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                   std::is_same_v<Policy, DCGAN>, double>
  Evaluate(const arma::mat& parameters,
           const size_t i,
           const size_t batchSize);

  /**
   * Evaluate function for the WGAN.
   * This function gives the performance of the WGAN on the current input.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, WGAN>, double>
  Evaluate(const arma::mat& parameters,
           const size_t i,
           const size_t batchSize);

  /**
   * Evaluate function for the WGAN-GP.
   * This function gives the performance of the WGAN-GP on the current input.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, WGANGP>, double>
  Evaluate(const arma::mat& parameters,
           const size_t i,
           const size_t batchSize);

  /**
   * EvaluateWithGradient function for the Standard GAN and DCGAN.
   * This function gives the performance of the Standard GAN or DCGAN on the
   * current input, while updating Gradients.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename GradType, typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                   std::is_same_v<Policy, DCGAN>, double>
  EvaluateWithGradient(const arma::mat& parameters,
                       const size_t i,
                       GradType& gradient,
                       const size_t batchSize);

  /**
   * EvaluateWithGradient function for the WGAN.
   * This function gives the performance of the WGAN on the
   * current input, while updating Gradients.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename GradType, typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, WGAN>, double>
  EvaluateWithGradient(const arma::mat& parameters,
                       const size_t i,
                       GradType& gradient,
                       const size_t batchSize);

  /**
   * EvaluateWithGradient function for the WGAN-GP.
   * This function gives the performance of the WGAN-GP on the
   * current input, while updating Gradients.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename GradType, typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, WGANGP>, double>
  EvaluateWithGradient(const arma::mat& parameters,
                       const size_t i,
                       GradType& gradient,
                       const size_t batchSize);

  /**
   * Gradient function for Standard GAN and DCGAN.
   * This function passes the gradient based on which network is being
   * trained, i.e., Generator or Discriminator.
   *
   * @param parameters present parameters of the network.
   * @param i Index of the predictors.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, StandardGAN> ||
                   std::is_same_v<Policy, DCGAN>, void>
  Gradient(const arma::mat& parameters,
           const size_t i,
           arma::mat& gradient,
           const size_t batchSize);

  /**
   * Gradient function for WGAN.
   * This function passes the gradient based on which network is being
   * trained, i.e., Generator or Discriminator.
   *
   * @param parameters present parameters of the network.
   * @param i Index of the predictors.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, WGAN>, void>
  Gradient(const arma::mat& parameters,
           const size_t i,
           arma::mat& gradient,
           const size_t batchSize);

  /**
   * Gradient function for WGAN-GP.
   * This function passes the gradient based on which network is being
   * trained, i.e., Generator or Discriminator.
   *
   * @param parameters present parameters of the network.
   * @param i Index of the predictors.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  template<typename Policy = PolicyType>
  std::enable_if_t<std::is_same_v<Policy, WGANGP>, void>
  Gradient(const arma::mat& parameters,
           const size_t i,
           arma::mat& gradient,
           const size_t batchSize);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  /**
   * This function does a forward pass through the GAN network.
   *
   * @param input Sampled noise.
   */
  void Forward(const arma::mat& input);

  /**
   * This function predicts the output of the network on the given input.
   *
   * @param input The input of the Generator network.
   * @param output Result of the Discriminator network.
   */
  void Predict(arma::mat input, arma::mat& output);

  //! Return the parameters of the network.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the parameters of the network.
  arma::mat& Parameters() { return parameter; }

  //! Return the generator of the GAN.
  const Model& Generator() const { return generator; }
  //! Modify the generator of the GAN.
  Model& Generator() { return generator; }
  //! Return the discriminator of the GAN.
  const Model& Discriminator() const { return discriminator; }
  //! Modify the discriminator of the GAN.
  Model& Discriminator() { return discriminator; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Get the matrix of responses to the input data points.
  const arma::mat& Responses() const { return responses; }
  //! Modify the matrix of responses to the input data points.
  arma::mat& Responses() { return responses; }

  //! Get the matrix of data points (predictors).
  const arma::mat& Predictors() const { return predictors; }
  //! Modify the matrix of data points (predictors).
  arma::mat& Predictors() { return predictors; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
  * Reset the module status by setting the current deterministic parameter
  * for the discriminator and generator networks and their respective layers.
  */
  void ResetDeterministic();

  //! Locally stored parameter for training data + noise data.
  arma::mat predictors;
  //! Locally stored parameters of the network.
  arma::mat parameter;
  //! Locally stored Generator network.
  Model generator;
  //! Locally stored Discriminator network.
  Model discriminator;
  //! Locally stored Initializer.
  InitializationRuleType initializeRule;
  //! Locally stored Noise function
  Noise noiseFunction;
  //! Locally stored input dimension of the Generator network.
  size_t noiseDim;
  //! Locally stored number of data points.
  size_t numFunctions;
  //! Locally stored batch size parameter.
  size_t batchSize;
  //! Locally stored batch number which is being processed.
  size_t currentBatch;
  //! Locally stored number of training step before Generator is trained.
  size_t generatorUpdateStep;
  //! Locally stored number of pre-train step for Discriminator.
  size_t preTrainSize;
  //! Locally stored learning rate ratio for Generator network.
  double multiplier;
  //! Locally stored weight clipping parameter.
  double clippingParameter;
  //! Locally stored lambda parameter.
  double lambda;
  //! Locally stored reset parameter.
  bool reset;
  //! Locally stored delta visitor.
  DeltaVisitor deltaVisitor;
  //! Locally stored responses.
  arma::mat responses;
  //! Locally stored current input.
  arma::mat currentInput;
  //! Locally stored current target.
  arma::mat currentTarget;
  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;
  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;
  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;
  //! Locally stored gradient parameters.
  arma::mat gradient;
  //! Locally stored gradient for Discriminator.
  arma::mat gradientDiscriminator;
  //! Locally stored gradient for noise data in the predictors.
  arma::mat noiseGradientDiscriminator;
  //! Locally stored norm of the gradient of Discriminator.
  arma::mat normGradientDiscriminator;
  //! Locally stored noise using the noise function.
  arma::mat noise;
  //! Locally stored gradient for Generator.
  arma::mat gradientGenerator;
  //! The current evaluation mode (training or testing).
  bool deterministic;
  //! To keep track of number of generator weights in total weights.
  size_t genWeights;
  //! To keep track of number of discriminator weights in total weights.
  size_t discWeights;
};

} // namespace mlpack

// Include implementation.
#include "gan_impl.hpp"
#include "wgan_impl.hpp"
#include "wgangp_impl.hpp"


#endif
