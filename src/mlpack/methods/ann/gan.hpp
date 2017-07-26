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
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>
#include <mlpack/methods/ann/visitor/reset_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_size_visitor.hpp>
#include <mlpack/methods/ann/visitor/weight_set_visitor.hpp>

#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::math;
using namespace mlpack::distribution;

namespace mlpack {
namespace ann /** Restricted Boltzmann Machine.  */ {
template<
typename Generator = FFN<>,
typename Discriminator = FFN<>,
typename IntializerType = RandomInitialization>
class GenerativeAdversarialNetwork
{
 public:
  GenerativeAdversarialNetwork(arma::mat& trainData,
      IntializerType initializeRule,
      Generator& generator,
      Discriminator& discriminator,
      size_t batchSize,
      size_t iterations,
      size_t disIteration,
      size_t generatorInSize);

  // Reset function
  void Reset();

  // Generate data for generator and discriminator
  void GenerateData(arma::mat& batchData, arma::mat& batchResponses,
      size_t offset);

  // Train function
  template<typename OptimizerType>
  void Train(OptimizerType& Optimizer);

  // Evaluate function
  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const bool deterministic = true);
  /**
   * Gradient function 
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
   * @param input  the noise input
   */
  void Forward(arma::mat&& input);

  /**
   * This function predicts the output of the network
   * on the given input
   *
   * @param input  the input  the discriminator network
   * @param output result of the discriminator network
   */
  void Predict(arma::mat&& input, arma::mat& output);

  /**
   * Generate function generates random noise 
   * samples from a given distribution with 
   * given args. Samples are stored in a local variable.
   *
   * @tparam NoiseFunction the distribution to sample from
   * @tparam Args the arguments types for args of the distribution
   * @param numSamples number of samples to be generated from the distribution
   * @param args the aruments of the distribution to samples from
   */
  void Generate(arma::mat&& fakeData, arma::mat&& noiseData);
  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }
  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }
  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally stored train data
  arma::mat& trainData;
  //! Locally stored Intialiser
  IntializerType  initializeRule;
  //! Locally stored parameters of the network
  arma::mat parameter;
  //! Locally stored generator
  Generator& generator;
  //! Locally stored discriminator
  Discriminator& discriminator;
  //! Locally stored number of data points
  size_t numFunctions;
  //! Locally stored trainGenerator parmaeter
  bool trainGenerator;
  //! Locally stored batch size parameter
  size_t batchSize;
  //! Locally stored number of iterations
  size_t iterations;
  //! Locally stored number of iterations of discriminator
  size_t disIteration;
  //! Locally stored input size for generator
  size_t generatorInSize;
  //! Locally stored reset parmaeter
  bool reset;
  //! Locally stored delta visitor
  DeltaVisitor deltaVisitor;

  //! Locally stored parameter for training data
  arma::mat predictors;
  //! Locally stored responses
  arma::mat responses;
  //! Locally stored current input
  arma::mat currentInput;
  //! Locally stored current target
  arma::mat currentTarget;

  //! Locally stored noise samples
  arma::mat noiseData;
  //! Locally stored data fake + real
  arma::mat data;
  //! Locally stored labels fake + real
  arma::mat labels;
  //! Locally stored discriminator fake Data
  arma::mat disFakeData;
  //! Locally stored generator fake data
  arma::mat genFakeData;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;
  //! Locally-stored weight size visitor.
  WeightSizeVisitor weightSizeVisitor;
  //! Locally-stored reset visitor.
  ResetVisitor resetVisitor;

  //! Locally stored gradient parameters
  arma::mat gradient;
  //! Locally stored gradient for discriminator
  arma::mat gradientDiscriminator;
  //! Locally stored gradient for generator
  arma::mat gradientGenerator;
  //! Locally stored output of the generator network
  arma::mat ganOutput;
};
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "gan_impl.hpp"

#endif
