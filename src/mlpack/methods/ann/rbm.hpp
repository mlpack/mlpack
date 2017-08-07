/**
 * @file rbm.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_HPP
#define MLPACK_METHODS_ANN_RBM_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

namespace mlpack {
namespace ann /** Artificial neural networks.  */ {

template<typename InitializationRuleType, typename RBMPolicy>
class RBM
{
 public:
  using NetworkType = RBM<InitializationRuleType, RBMPolicy>;

  /* 
   * Intalise all the parameters of the network
   * using the intialise rule. 
   *
   * @tparam IntialiserType rule to intialise the parameters of the network
   * @param predictors training data
   * @param numSteps Number of gibbs steps sampling
   * @param useMonitoringCost evaluation function to use
   * @param persistence indicates to use persistent CD
   */
  RBM(arma::mat predictors, InitializationRuleType initializeRule,
      RBMPolicy rbmPolicy,
      const size_t numSteps = 1,
      const size_t mSteps = 1,
      const bool useMonitoringCost = true,
      const bool persistence = false);

  // Reset the network
  void Reset();

  /* 
   * Train the network using the Opitimzer with given set of args.
   * the optimiser sets the parameters of the network for providing
   * most likely parameters given the inputs
   * @param: predictors data points
   * @param: optimizer Optimizer type
   */
  template<typename OptimizerType>
  void Train(const arma::mat& predictors, OptimizerType& optimizer);

 /**
  * Evaluate the rbm network with the given parameters.
  * The function is needed for monitoring the progress of the network.
  *
  * @param parameters Matrix model parameters.
  * @param i Index of point to use for objective function evaluation.
  */
  double Evaluate(const arma::mat& parameters, const size_t i);

 /** 
  * This function calculates
  * the free energy of the model
  * @param: input data point 
  */
  double FreeEnergy(arma::mat&& input);

 /*
  * This functions samples the hidden
  * layer given the visible layer
  *
  * @param input visible layer input
  * @param output the sampled hidden layer
  */
  void SampleHidden(arma::mat&& input, arma::mat&& output);

  /*
  * This functions samples the visible
  * layer given the hidden layer
  *
  * @param input hidden layer
  * @param output the sampled visible layer 
  */
  void SampleVisible(arma::mat&& input, arma::mat&& output);

 /*
  * This function does the k-step
  * gibbs sampling.
  *
  * @param input: input to the gibbs function
  * @param output: stores the negative sample
  * @param steps: number of gibbs sampling steps
  */
  void Gibbs(arma::mat&& input, arma::mat&& output, size_t steps = SIZE_MAX);

  /*
   * Calculates the gradients for the rbm network
   *
   * @param parameters the current parmaeters of the network
   * @param input index the visible layer/data point
   * @param output store the gradients
   */
  void Gradient(arma::mat& parameters, const size_t input, arma::mat& output);

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the number of stes of gibbs sampling.
  size_t NumSteps() const { return numSteps; }

  //! Return the parameters of the network
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the parameters of the network
  arma::mat& Parameters() { return parameter; }

  //! Retutrn the rbm policy for the network
  const RBMPolicy& Policy() const { return rbmPolicy; }
  //! Modify the rbm policy for the network
  RBMPolicy& Policy() { return rbmPolicy; }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally stored parameters of the network
  arma::mat parameter;
  //! Policy type of RBM
  RBMPolicy rbmPolicy;
  //! The matrix of data points (predictors).
  arma::mat predictors;
  // Intialiser
  InitializationRuleType initializeRule;
  //! Locally-stored state of the persistent cdk.
  arma::mat state;
  //! Locally-stored number of data points
  size_t numFunctions;
  //! Locally-stored number of steps in gibbs sampling
  size_t numSteps;
  //! Locally-stored number of negative samples
  size_t mSteps;
  //! Locally-stored monitoring cost
  bool useMonitoringCost;
  //! Locally-stored persistent cd-k or not
  bool persistence;
  //! Locally-stored reset variable
  bool reset;

  //! Locally-stored reconstructed output from hidden layer
  arma::mat hiddenReconstruction;
  //! Locally-stored reconstructed output from visible layer
  arma::mat visibleReconstruction;

  //! Locally-stored negative samples from gibbs Distribution
  arma::mat negativeSamples;
  //! Locally-stored gradients from the negative phase
  arma::mat negativeGradient;
  //! Locally-stored temproray negative gradient used for negative phase
  arma::mat tempNegativeGradient;
  //! Locally-stored gradient for positive phase
  arma::mat positiveGradient;
  //! Locally-stored temporary output of gibbs chain
  arma::mat gibbsTemporary;
  //! Locally-stored output of the preActivation function used in FreeEnergy
  arma::mat preActivation;
};
} // namespace ann
} // namespace mlpack

#include "rbm_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_HPP
