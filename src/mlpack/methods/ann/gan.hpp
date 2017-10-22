/**
 * @file gan.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
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
namespace ann /** artifical neural network **/ {
template<typename Model, typename InitializationRuleType, class Noise>
class GAN
{
 public:
  /**
   * Constructor for GAN class
   *
   * @tparam Model The class type of generator and discriminator.
   
   * @param trainData The real data.
   * @param generator Generator network.
   * @param discriminator Discriminator network.
   * @param InitializationRuleType Type of Intializer.
   * @param noiseFunction The noise function to use.
   * @param batchSize BatchSize to be used for training.
   * @param generatorUpdateStep Num of steps of discriminator training before 
   *                            updating generator.
   * @param preTrainSize Num of preTraining step of discriminator.
   * @parma multiplier Ratio of learning rate of discriminator to the generator.
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

  // Reset function
  void Reset();

  // Train function
  template<typename OptimizerType>
  void Train(OptimizerType& Optimizer);

  /**
   * Evaluate function for the GAN
   * gives the perfomance of the gan
   * on the current input.
   *
   * @param parameters The parameters of the network
   * @param i The idx of the current input
   */
  double Evaluate(const arma::mat& parameters, const size_t i);

  /**
   * Gradient function for gan. 
   * This function is passes the gradient based
   * on which network is being trained ie generator or Discriminator.
   * 
   * @param parameters present parameters of the network
   * @param i index of the predictors
   * @param gradient variable to store the present gradient
   */
  void Gradient(const arma::mat& parameters, const size_t i,
      arma::mat& gradient);

  /**
   * This function does forward pass through the GAN
   * network.
   *
   * @param input Sampled noise
   */
  void Forward(arma::mat&& input);

  /**
   * This function predicts the output of the network
   * on the given input.
   *
   * @param input  the input  the discriminator network
   * @param output result of the discriminator network
   */
  void Predict(arma::mat&& input, arma::mat& output);

  //! Return the parameters of the network.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the parameters of the network
  arma::mat& Parameters() { return parameter; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally stored parameter for training data + noise data.
  arma::mat predictors;
  //! Locally stored parameters of the network.
  arma::mat parameter;
  //! Locally stored generator network.
  Model& generator;
  //! Locally stored discriminator network.
  Model& discriminator;
  //! Locally stored Intialiser.
  InitializationRuleType  initializeRule;
  //! Locally stored Noise function
  Noise noiseFunction;
  //! Locally stored input dimension of the generator network.
  size_t noiseDim;
  //! Locally stored number of data points.
  size_t numFunctions;
  //! Locally stored batch size parameter.
  size_t batchSize;
  //! Locally stored number of iterations that have been completed.
  size_t counter;
  //! Locally stored batch number which is being processed.
  size_t currentBatch;
  //! Locally stored number of training step before generator is trained.
  size_t generatorUpdateStep;
  //! Locally stored number of pre-train step for discriminator.
  size_t preTrainSize;
  //! Locally stored learning rate ratio for generator network.
  double multiplier;
  //! Locally stored reset parmaeter.
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
  //! Locally stored gradient for discriminator.
  arma::mat gradientDiscriminator;
  //! Locally stored gradient for noise data in the predictors.
  arma::mat noiseGradientDiscriminator;
  //! Locally stored noise using the noise function.
  arma::mat noise;
  //! Locally stored gradient for generator.
  arma::mat gradientGenerator;
  //! Locally stored output of the generator network.
  arma::mat ganOutput;
};
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gan_impl.hpp"

#endif
