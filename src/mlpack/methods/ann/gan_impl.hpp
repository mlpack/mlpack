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
typename Generator, 
typename Discriminator,
typename NoiseFunction,
typename IntializerType,>
 GenerativeAdversarialNetwork::GenerativeAdversarialNetwork(arma::mat trainData,
      arma::mat trainLables,
      IntializerType initializeRule,
      Generator& generator,
      Discriminator& discriminator,
      NoiseFunction& noise,
      size_t batchSize):
      initializeRule(initializeRule),
      generator(generator),
      discriminator(discriminator),
      noise(noise),
      trainGenerator(false),
      batchSize(batchSize),
      reset(false)
      {
        numFunctions = trainData.n_cols;
        predictors = trainData;
        responses = trainLables;
        fakeData.set_size(trainData.n_rows, batchSize);
      };
template<
typename Generator, 
typename Discriminator,
typename NoiseFunction,
typename IntializerType>
 void GenerativeAdversarialNetwork::Reset()
  {
    // Call the reset function of both the Generator and Discriminator Network
    generator.ResetParameters();
    discriminator.ResetParameters();
    fakeData.zeros();
    fakeLables.zeros();
    parameter.set_size(generator.Parameters().n_rows + 
        discriminator.Parameters().n_rows,
        generator.Parameters().n_cols + generator.Parameters().n_cols);
    initializeRule.Initialize(parameter, parameter.n_elem, 1);
    generator.Parameters() = arma::mat(parameter.memptr(),
        generator.Parameters().n_rows, generator.Parameters().n_cols,
        false, false);
    discriminator.Parameters() = arma::mat(parameter.memptr(),
      discriminator.Parameters().n_rows, discriminator.Parameters().n_cols,
      false, false);
  }
template<
  typename Generator, 
  typename Discriminator,
  typename NoiseFunction,
  typename IntializerType>
  template<typename OptimizerType>
 void GenerativeAdversarialNetwork::Train(OptimizerType& Optimizer)
  {
    if (!reset)
      Reset();
    size_t offset = 0;
    for (size_t i = 0; i< predictors.n_cols / batchSize; i++)
    {
      Generate(batchSize, std::move(fakeData));
      fakeLables = arma::zeros(1, predictors.n_cols);
      tempTrainData.set_size(predictors.n_rows, 2 * batchSize);
      tempLabels.set_size(1, 2 * batchSize);
      tempTrainData.cols(0, batchSize - 1) = arma::mat(
        fakeData.memptr() + offset, fakeData.n_rows, batchSize, false, false);
      tempTrainData.cols(batchSize, tempTrainData.n_cols - 1) = arma::mat(
          predictors.memptr() + offset, predictors.n_rows, batchSize,
          false, false);
      tempLabels.cols(0, batchSize - 1) = arma::mat(
          fakeLables.memptr() + offset, fakeLables.n_rows, batchSize,
          false, false);
      tempLabels.cols(batchSize, tempLabels.n_cols - 1) = arma::mat(
          responses.memptr() + offset,responses.n_rows, batchSize,
          false, false);
      offset += batchSize;
      // Train the discrminator network
      this->predictors = std::move(tempTrainData);
      this->responses = std::move(tempLabels);
      numFunctions = this->predictors.n_cols;
      std::cout << "numFunctions = " << numFunctions << std::endl;
      Optimizer.Optimize(*this, parameter);
      trainGenerator = true;
      // Train the generator network
      Generate(predictors.n_cols, std::move(fakeData));
      this->responses = arma::ones(responses.n_cols);
      numFunctions = predictors.n_cols;
      std::cout << "predictors.n_cols" << predictors.n_cols << std::endl;
      std::cout << "NumFunctions = " << numFunctions << std::endl;
      Optimizer.MaxIterations() *= 10;
      Optimizer.Optimize(*this, parameter);
    }
  }
template<typename Generator, typename Discriminator, typename NoiseFunction,
    typename IntializerType>
double GenerativeAdversarialNetwork::Evaluate(const arma::mat& parameters,
                  const size_t i,
                  const bool deterministic = true)
  {
    // Todo fix this
    Generate(size_t numSamples, arma::mat&& fakeData)
    generator.Forward(std::move(fakeData));
    boost::apply_visitor(outputParameterVisitor, generator.Network().back());
    return 1 - arma::log(discriminator.Forward(std::move(fakeData)));
  }

  /**
   * Gradient function 
   */
  void Gradient(const arma::mat& parameters, const size_t i,
      arma::mat& gradient)
  {
    if (gradient.is_empty())
    {
      if (parameter.is_empty())
      {
        Reset();
      }
    gradient = arma::zeros<arma::mat>(generator.Parameters().n_rows +
        discriminator.Parameters().n_rows, generator.Parameters().n_cols +
        discriminator.Parameters().n_rows);
    }
    else
    {
      gradient.zeros();
    }

    // Gradient for generator network
    gradientGenerator = arma::mat(gradient.memptr(),
        generator.Parameters().n_rows,
        generator.Parameters().n_cols, false, false);

    // Gradient for discriminator network
    gradientDisriminator = arma::mat(gradient.memptr(),
        discriminator.Parameters().n_rows,
        discriminator.Parameters().n_cols, false, false);

    // Get the discriminator gradient 
    discriminator.Gradient(parameters, i, gradientDisriminator);
    if (trainGenerator)
    {
      // Use visitors later
      boost::apply_visitor(DeltaVisitor(), discriminator.Network().front()) =
          generator.OutputLayer().Delta();
      generator.Gradient(parameters, i, gradientGenerator);
    }

    if (trainGenerator)
      gradientGenerator.zeros();
    else
      gradientDisriminator.zeros();
  }

  /**
   * This function does forward pass through the GAN
   * network.
   *
   * @param input  the noise input
   */
  void Forward(arma::mat&& input)
  {
    if (!reset)
      Reset();
    generator.Evaluate(std::move(input));
    ganOutput = boost::apply_visitor(outputParameterVisitor,
        generator.Network().back());
    return discriminator.Forward(std::move(ganOutput));
  }

  /**
   * This function predicts the output of the network
   * on the given input
   *
   * @param input  the input  the discriminator network
   * @param output result of the discriminator network
   */
  void Predict(arma::mat&& input, arma::mat& output)
  {
    if (!reset)
      Reset();
    discriminator.Forward(std::move(input));
    output = boost::apply_visitor(outputParameterVisitor,
        discriminator.Network().back());
  }

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
  void Generate(size_t numSamples, arma::mat&& fakeData)
  {
    for (size_t i = 0; i < numSamples; i++)
    {
      generator.Forward(std::move(noise.Random()));
      fakeData.col(i) = boost::apply_visitor(outputParameterVisitor,
          generator.Network().back());
    }
  }
};
}
}
# endif