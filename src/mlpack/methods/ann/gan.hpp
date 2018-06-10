/**
 * @file gan.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_GAN_HPP
#define MLPACK_METHODS_ANN_GAN_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>
#include <mlpack/methods/ann/visitor/reset_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_size_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_set_visitor.hpp>


namespace mlpack {
namespace ann /** Artificial Neural Network. **/ {

/**
 * The implementation of the standard GAN module. Generative Adversarial
 * Networks (GANs) are a class of artificial intelligence algorithms used
 * in unsupervised machine learning, implemented by a system of two neural
 * networks contesting with each other in a zero-sum game framework. This
 * technique can generate photographs that look at least superficially
 * authentic to human observers, having many realistic characteristics.
 *
 * For more information, see the following papers.
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
 * @code
 * @article{Salimans16,
 *   author    = {Tim Salimans, Ian Goodfellow, Wojciech Zaremba,
 *                Vicki Cheung, Alec Radford and Xi Chen},
 *   title     = {Improved Techniques for Training GANs},
 *   year      = {2016},
 *   url       = {http://arxiv.org/abs/1606.03498},
 *   eprint    = {1606.03498},
 * }
 * @endcode
 *
 * @tparam Model The class type of Generator and Discriminator.
 * @tparam InitializationRuleType Type of Initializer.
 * @tparam Noise The noise function to use.
 */
template<
  typename Model,
  typename InitializationRuleType,
  class Noise
>
class GAN
{
 public:
  /**
   * Constructor for GAN class.
   *
   * @param trainData The real data.
   * @param generator Generator network.
   * @param discriminator Discriminator network.
   * @param batchSize BatchSize to be used for training.
   * @param generatorUpdateStep Number of steps of Discriminator training
   *                            before updating generator.
   * @param preTrainSize Num of preTraining step of Discriminator.
   * @param multiplier Ratio of learning rate of Discriminator to the Generator.
   */
  GAN(arma::mat& trainData,
      Model& generator,
      Model& discriminator,
      InitializationRuleType initializeRule,
      Noise noiseFunction,
      size_t noiseDim,
      size_t batchSize,
      size_t generatorUpdateStep,
      size_t preTrainSize,
      double multiplier);

  // Reset function.
  void Reset();

  // Train function.
  template<typename OptimizerType>
  void Train(OptimizerType& Optimizer);

  /**
   * Evaluate function for the GAN gives the performance of the GAN on the
   * current input.
   *
   * @param parameters The parameters of the network.
   * @param i Index of the current input.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const size_t batchSize);

  /**
   * Gradient function for GAN.
   * This function passes the gradient based on which network is being
   * trained, i.e., Generator or Discriminator.
   * 
   * @param parameters present parameters of the network.
   * @param i Index of the predictors.
   * @param gradient Variable to store the present gradient.
   */
  void Gradient(const arma::mat& parameters,
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
  void Forward(arma::mat&& input);

  /**
   * This function predicts the output of the network on the given input.
   *
   * @param input The input the Discriminator network.
   * @param output Result of the Discriminator network.
   */
  void Predict(arma::mat&& input,
               arma::mat& output);

  //! Return the parameters of the network.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the parameters of the network
  arma::mat& Parameters() { return parameter; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally stored parameter for training data + noise data.
  arma::mat predictors;
  //! Locally stored parameters of the network.
  arma::mat parameter;
  //! Locally stored Generator network.
  Model& generator;
  //! Locally stored Discriminator network.
  Model& discriminator;
  //! Locally stored Initializer.
  InitializationRuleType  initializeRule;
  //! Locally stored Noise function
  Noise noiseFunction;
  //! Locally stored input dimension of the Generator network.
  size_t noiseDim;
  //! Locally stored number of data points.
  size_t numFunctions;
  //! Locally stored batch size parameter.
  size_t batchSize;
  //! Locally stored number of iterations that have been completed.
  size_t counter;
  //! Locally stored batch number which is being processed.
  size_t currentBatch;
  //! Locally stored number of training step before Generator is trained.
  size_t generatorUpdateStep;
  //! Locally stored number of pre-train step for Discriminator.
  size_t preTrainSize;
  //! Locally stored learning rate ratio for Generator network.
  double multiplier;
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
  //! Locally stored noise using the noise function.
  arma::mat noise;
  //! Locally stored gradient for Generator.
  arma::mat gradientGenerator;
  //! Locally stored output of the Generator network.
  arma::mat ganOutput;
};
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gan_impl.hpp"

#endif
