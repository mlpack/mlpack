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

#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack {
namespace ann /** Artificial neural networks.  */ {
/* 
 * RBM class
 *
 * @tparam IntialiserType rule to intialise the parameters of the network
 * @tparam RBMPolicy type of rbm
 */

template<typename InitializationRuleType, typename RBMPolicy>
class RBM
{
 public:
  using NetworkType = RBM<InitializationRuleType, RBMPolicy>;
  typedef typename RBMPolicy::ElemType ElemType;

  /* 
   * Intalise all the parameters of the network
   * using the intialise rule. 
   *
   * @tparam RbmPolicy Class of RBM to use(ssRBM / BinaryRBM).
   * @param predictors Training data to used.
   * @param numSteps Number of gibbs steps sampling.
   * @param negSteps Number of negative samples to average negative gradient.
   * @param useMonitoringCost Indicates whic Evaluation type to use.
   * @param persistence Indicates whether to use persistent CD or not.
   */
  RBM(arma::Mat<ElemType> predictors,
      InitializationRuleType initializeRule,
      RBMPolicy rbmPolicy,
      const size_t numSteps = 1,
      const size_t negSteps = 1,
      const bool useMonitoringCost = true,
      const bool persistence = false);

  // Reset the network
  void Reset();

  /* 
   * Train the feedforward network on the given input data.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @param predictors Data points / Traing Data.
   * @param optimizer Optimizer type.
   */
  template<typename OptimizerType>
  void Train(const arma::Mat<ElemType>& predictors, OptimizerType& optimizer);

 /**
  * Evaluate the rbm network with the given parameters.
  * The function is needed for monitoring the progress of the network.
  *
  * @param parameters Matrix model parameters.
  * @param i Index of point to use for objective function evaluation.
  */
  double Evaluate(const arma::Mat<ElemType>& parameters, const size_t i);

 /** 
  * This function calculates the free energy of the model.
  *
  * @param Input data point.
  */
  double FreeEnergy(arma::Mat<ElemType>&& input);

 /*
  * This functions samples the hidden layer given the visible layer.
  *
  * @param input Visible layer input.
  * @param output The sampled hidden layer.
  */
  void SampleHidden(arma::Mat<ElemType>&& input, arma::Mat<ElemType>&& output);

  /*
  * This functions samples the visible layer given the hidden layer.
  *
  * @param input Hidden layer of the network.
  * @param output The sampled visible layer.
  */
  void SampleVisible(arma::Mat<ElemType>&& input, arma::Mat<ElemType>&& output);

 /*
  * This function does the k-step gibbs sampling.
  *
  * @param input Input to the gibbs function.
  * @param output Used for storing the negative sample.
  * @param steps Number of gibbs sampling steps taken.
  */
  void Gibbs(arma::Mat<ElemType>&& input,
             arma::Mat<ElemType>&& output,
             size_t steps = SIZE_MAX);

  /*
   * Calculates the gradients for the rbm network.
   *
   * @param parameters The current parameters of the network.
   * @param input Index of the visible layer/data point.
   * @param output Used for storing the gradients.
   */
  void Gradient(arma::Mat<ElemType>& parameters,
                const size_t input,
                arma::Mat<ElemType>& output);

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the number of stes of gibbs sampling.
  size_t NumSteps() const { return numSteps; }

  //! Return the parameters of the network.
  const arma::Mat<ElemType>& Parameters() const { return parameter; }
  //! Modify the parameters of the network.
  arma::Mat<ElemType>& Parameters() { return parameter; }

  //! Retutrn the rbm policy for the network.
  const RBMPolicy& Policy() const { return rbmPolicy; }
  //! Modify the rbm policy for the network.
  RBMPolicy& Policy() { return rbmPolicy; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally stored parameters of the network.
  arma::Mat<ElemType> parameter;
  //! Policy type of RBM.
  RBMPolicy rbmPolicy;
  //! The matrix of data points (predictors).
  arma::Mat<ElemType> predictors;
  // Intialiser for intializing the weights of the network.
  InitializationRuleType initializeRule;
  //! Locally-stored state of the persistent cdk.
  arma::Mat<ElemType> state;
  //! Locally-stored number of data points.
  size_t numFunctions;
  //! Locally-stored number of steps in gibbs sampling.
  size_t numSteps;
  //! Locally-stored number of negative samples.
  size_t negSteps;
  //! Locally-stored monitoring cost.
  bool useMonitoringCost;
  //! Locally-stored persistent cd-k or not.
  bool persistence;
  //! Locally-stored reset variable.
  bool reset;

  //! Locally-stored reconstructed output from hidden layer.
  arma::Mat<ElemType> hiddenReconstruction;
  //! Locally-stored reconstructed output from visible layer.
  arma::Mat<ElemType> visibleReconstruction;

  //! Locally-stored negative samples from gibbs distribution.
  arma::Mat<ElemType> negativeSamples;
  //! Locally-stored gradients from the negative phase.
  arma::Mat<ElemType> negativeGradient;
  //! Locally-stored temproray negative gradient used for negative phase.
  arma::Mat<ElemType> tempNegativeGradient;
  //! Locally-stored gradient for positive phase.
  arma::Mat<ElemType> positiveGradient;
  //! Locally-stored temporary output of gibbs chain.
  arma::Mat<ElemType> gibbsTemporary;
  //! Locally-stored output of the preActivation function used in FreeEnergy.
  arma::Mat<ElemType> preActivation;
};
} // namespace ann
} // namespace mlpack

#include "rbm_impl.hpp"

#endif // MLPACK_METHODS_ANN_RBM_HPP
