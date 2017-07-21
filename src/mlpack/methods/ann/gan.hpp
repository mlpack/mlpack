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
  GenerativeAdversarialNetwork(arma::mat trainData, arma::mat trainLables,
      IntializerType initializeRule,
      Generator& generator,
      Discriminator& discriminator,
      size_t batchSize,
      size_t generatorInSize);

  void Reset();

  template<typename OptimizerType>
  void Train(OptimizerType& Optimizer);

  double Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const bool deterministic = true);
  /**
   * Gradient function 
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
  void Generate(size_t numSamples, arma::mat&& fakeData, arma::mat&& noiseData);
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
  //! Locally stored fake data used for training
  arma::mat fakeData;
  //! Locally stored noise samples
  arma::mat noiseData;
  //! Locally stored fake Labels used for training
  arma::mat fakeLables;
  //! Locally stored train data comprising of real and fake data
  arma::mat tempTrainData;
  //! Locally stored temp variable comprisiong of read and fake labels
  arma::mat tempLabels;
  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;
  //! Locally stored gradient parameters
  arma::mat gradient;
  //! Locally stored gradient for discriminator
  arma::mat gradientDisriminator;
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
