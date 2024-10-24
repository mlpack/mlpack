/**
 * @file methods/ann/rbm/rbm.hpp
 * @author Kris Singh
 * @author Shikhar Jaiswal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_RBM_RBM_HPP
#define MLPACK_METHODS_ANN_RBM_RBM_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/rbm/rbm_policies.hpp>

namespace mlpack {

/**
 * The implementation of the RBM module. A Restricted Boltzmann Machines (RBM)
 * is a generative stochastic artificial neural network that can learn a
 * probability distribution over its set of inputs. RBMs have found applications
 * in dimensionality reduction, classification, collaborative filtering, feature
 * learning and topic modelling. They can be trained in either supervised or
 * unsupervised ways, depending on the task. They are a variant of Boltzmann
 * machines, with the restriction that the neurons must form a bipartite graph.
 *
 * @tparam InitializationRuleType Rule used to initialize the network.
 * @tparam DataType The type of matrix to be used.
 * @tparam PolicyType The RBM variant to be used (BinaryRBM or SpikeSlabRBM).
 */
template<
  typename InitializationRuleType,
  typename DataType = arma::mat,
  typename PolicyType = BinaryRBM
>
class RBM
{
 public:
  using NetworkType = RBM<InitializationRuleType, DataType, PolicyType>;
  using ElemType = typename DataType::elem_type;

  /**
   * Initialize all the parameters of the network using initializeRule.
   *
   * @param predictors Training data to be used.
   * @param initializeRule InitializationRule object for initializing the
   *        network parameter.
   * @param visibleSize Number of visible neurons.
   * @param hiddenSize Number of hidden neurons.
   * @param batchSize Batch size to be used for training.
   * @param numSteps Number of Gibbs Sampling steps.
   * @param negSteps Number of negative samples to average negative gradient.
   * @param poolSize Number of hidden neurons to pool together.
   * @param slabPenalty Regulariser of slab variables.
   * @param radius Feasible regions for visible layer samples.
   * @param persistence Indicates whether to use Persistent CD or not.
   */
  RBM(arma::Mat<ElemType> predictors,
      InitializationRuleType initializeRule,
      const size_t visibleSize,
      const size_t hiddenSize,
      const size_t batchSize = 1,
      const size_t numSteps = 1,
      const size_t negSteps = 1,
      const size_t poolSize = 2,
      const ElemType slabPenalty = 8,
      const ElemType radius = 1,
      const bool persistence = false);

  // Reset the network.
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, BinaryRBM>, void>
  Reset();

  // Reset the network.
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  Reset();

  /**
   * Train the RBM on the given input data.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback functions.
   * @param optimizer Optimizer type.
   * @param callbacks Callback Functions for ensmallen optimizer
   *      `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackType>
  double Train(OptimizerType& optimizer, CallbackType&&... callbacks);

  /**
   * Evaluate the RBM network with the given parameters.
   * The function is needed for monitoring the progress of the network.
   *
   * @param parameters Matrix model parameters.
   * @param i Index of the data point.
   * @param batchSize Variable to store the present number of inputs.
   */
  double Evaluate(const arma::Mat<ElemType>& parameters,
                  const size_t i,
                  const size_t batchSize);

  /**
   * This function calculates the free energy of the BinaryRBM.
   * The free energy is given by:
   * @f$ -b^Tv - \sum_{i=1}^M log(1 + e^{c_j+v^TW_j}) @f$.
   *
   * @param input The visible neurons.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, BinaryRBM>, double>
  FreeEnergy(const arma::Mat<ElemType>& input);

  /**
   * This function calculates the free energy of the SpikeSlabRBM.
   * The free energy is given by:
   * @f$ v^t$$\Delta$v - $\sum_{i=1}^N @f$
   * @f$ \log{ \sqrt{\frac{(-2\pi)^K}{\prod_{m=1}^{K}(\alpha_i)_m}}} @f$ -
   * @f$ \sum_{i=1}^N \log(1+\exp( b_i +
   * \sum_{m=1}^k \frac{(v(w_i)_m^t)^2}{2(\alpha_i)_m}) @f$
   *
   * @param input The visible layer neurons.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, double>
  FreeEnergy(const arma::Mat<ElemType>& input);

  /**
   * Calculates the gradient of the RBM network on the provided input.
   *
   * @param input The provided input data.
   * @param gradient Stores the gradient of the RBM network.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, BinaryRBM>, void>
  Phase(const InputType& input, DataType& gradient);

  /**
   * Calculates the gradient of the RBM network on the provided input.
   *
   * @param input The provided input data.
   * @param gradient Stores the gradient of the RBM network.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  Phase(const InputType& input, DataType& gradient);

  /**
   * This function samples the hidden layer given the visible layer using
   * Bernoulli function.
   *
   * @param input Visible layer input.
   * @param output The sampled hidden layer.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, BinaryRBM>, void>
  SampleHidden(const arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);

  /**
   * This function samples the slab outputs from the Normal distribution with
   * mean given by:
   * @f$ h_i*\alpha^{-1}*W_i^T*v @f$
   * and variance:
   * @f$ \alpha^{-1} @f$
   *
   * @param input Consists of both visible and spike variables.
   * @param output Sampled slab neurons.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  SampleHidden(const arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);

  /**
   * This function samples the visible layer given the hidden layer using
   * Bernoulli function.
   *
   * @param input Hidden layer of the network.
   * @param output The sampled visible layer.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, BinaryRBM>, void>
  SampleVisible(arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);

  /**
   * Sample Hidden function samples the slab outputs from the Normal
   * distribution with mean given by:
   * @f$ h_i*\alpha^{-1}*W_i^T*v @f$
   * and variance:
   * @f$ \alpha^{-1} @f$
   *
   * @param input Hidden layer of the network.
   * @param output The sampled visible layer.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  SampleVisible(arma::Mat<ElemType>& input, arma::Mat<ElemType>& output);

  /**
   * The function calculates the mean for the visible layer.
   *
   * @param input Hidden neurons from the hidden layer of the network.
   * @param output Visible neuron activations.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, BinaryRBM>, void>
  VisibleMean(InputType& input, DataType& output);

  /**
   * The function calculates the mean of the Normal distribution of P(v|s, h).
   * The mean is given by:
   * @f$ \Lambda^{-1} \sum_{i=1}^N W_i * s_i * h_i @f$
   *
   * @param input Consists of both the spike and slab variables.
   * @param output Mean of the of the Normal distribution.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  VisibleMean(InputType& input, DataType& output);

  /**
   * The function calculates the mean for the hidden layer.
   *
   * @param input Visible neurons.
   * @param output Hidden neuron activations.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, BinaryRBM>, void>
  HiddenMean(const InputType& input, DataType& output);

  /**
   * The function calculates the mean of the Normal distribution of P(s|v, h).
   * The mean is given by:
   * @f$ h_i*\alpha^{-1}*W_i^T*v @f$
   * The variance is given by:
   * @f$ \alpha^{-1} @f$
   *
   * @param input Visible layer neurons.
   * @param output Consists of both the spike samples and slab samples.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  HiddenMean(const InputType& input, DataType& output);

  /**
   * The function calculates the mean of the distribution P(h|v),
   * where mean is given by:
   * @f$ sigm(v^T*W_i*\alpha_i^{-1}*W_i^T*v + b_i) @f$
   *
   * @param visible The visible layer neurons.
   * @param spikeMean Indicates P(h|v).
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  SpikeMean(const InputType& visible, DataType& spikeMean);

  /**
   * The function samples the spike function using Bernoulli distribution.
   * @param spikeMean Indicates P(h|v).
   * @param spike Sampled binary spike variables.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  SampleSpike(InputType& spikeMean, DataType& spike);

  /**
   * The function calculates the mean of Normal distribution of P(s|v, h),
   * where the mean is given by:
   * @f$ h_i*\alpha^{-1}*W_i^T*v @f$
   *
   * @param visible The visible layer neurons.
   * @param spike The spike variables from hidden layer.
   * @param slabMean The mean of the Normal distribution of slab neurons.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  SlabMean(const DataType& visible, DataType& spike, DataType& slabMean);

  /**
   * The function samples from the Normal distribution of P(s|v, h),
   * where the mean is given by:
   * @f$ h_i*\alpha^{-1}*W_i^T*v @f$
   * and variance is given by:
   * @f$ \alpha^{-1} @f$
   *
   * @param slabMean Mean of the Normal distribution of the slab neurons.
   * @param slab Sampled slab variable from the Normal distribution.
   */
  template<typename Policy = PolicyType, typename InputType = DataType>
  std::enable_if_t<std::is_same_v<Policy, SpikeSlabRBM>, void>
  SampleSlab(InputType& slabMean, DataType& slab);

  /**
   * This function does the k-step Gibbs Sampling.
   *
   * @param input Input to the Gibbs function.
   * @param output Used for storing the negative sample.
   * @param steps Number of Gibbs Sampling steps taken.
   */
  void Gibbs(const arma::Mat<ElemType>& input,
             arma::Mat<ElemType>& output,
             const size_t steps = SIZE_MAX);

  /**
   * Calculates the gradients for the RBM network.
   *
   * @param parameters The current parameters of the network.
   * @param i Index of the data point.
   * @param gradient Variable to store the present gradient.
   * @param batchSize Variable to store the present number of inputs.
   */
  void Gradient(const arma::Mat<ElemType>& parameters,
                const size_t i,
                arma::Mat<ElemType>& gradient,
                const size_t batchSize);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  //! Return the number of steps of Gibbs Sampling.
  size_t NumSteps() const { return numSteps; }

  //! Return the parameters of the network.
  const arma::Mat<ElemType>& Parameters() const { return parameter; }
  //! Modify the parameters of the network.
  arma::Mat<ElemType>& Parameters() { return parameter; }

  //! Get the weights of the network.
  arma::Cube<ElemType> const& Weight() const { return weight; }
  //! Modify the weights of the network.
  arma::Cube<ElemType>& Weight() { return weight; }

  //! Return the visible bias of the network.
  DataType const& VisibleBias() const { return visibleBias; }
  //! Modify the visible bias of the network.
  DataType& VisibleBias() { return visibleBias; }

  //! Return the hidden bias of the network.
  DataType const& HiddenBias() const { return hiddenBias; }
  //! Modify the  hidden bias of the network.
  DataType& HiddenBias() { return hiddenBias; }

  //! Get the regularizer associated with spike variables.
  DataType const& SpikeBias() const { return spikeBias; }
  //! Modify the regularizer associated with spike variables.
  DataType& SpikeBias() { return spikeBias; }

  //! Get the regularizer associated with slab variables.
  ElemType const& SlabPenalty() const { return 1.0 / slabPenalty; }

  //! Get the regularizer associated with visible variables.
  DataType const& VisiblePenalty() const { return visiblePenalty; }
  //! Modify the regularizer associated with visible variables.
  DataType& VisiblePenalty() { return visiblePenalty; }

  //! Get the visible size.
  size_t const& VisibleSize() const { return visibleSize; }
  //! Get the hidden size.
  size_t const& HiddenSize() const { return hiddenSize; }
  //! Get the pool size.
  size_t const& PoolSize() const { return poolSize; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! Locally stored parameters of the network.
  arma::Mat<ElemType> parameter;
  //! The matrix of data points (predictors).
  arma::Mat<ElemType> predictors;
  // Initializer for initializing the weights of the network.
  InitializationRuleType initializeRule;
  //! Locally-stored state of the persistent CD-k.
  arma::Mat<ElemType> state;
  //! Locally-stored number of data points.
  size_t numFunctions;
  //! Locally stored number of visible neurons.
  size_t visibleSize;
  //! Locally stored number of hidden neurons.
  size_t hiddenSize;
  //! Locally stored batch size parameter.
  size_t batchSize;
  //! Locally-stored number of steps in Gibbs Sampling.
  size_t numSteps;
  //! Locally-stored number of negative samples.
  size_t negSteps;
  //! Locally stored variable poolSize.
  size_t poolSize;
  //! Locally stored number of Sampling steps.
  size_t steps;
  //! Locally stored weight of the network.
  arma::Cube<ElemType> weight;
  //! Locally stored biases of the visible layer.
  DataType visibleBias;
  //! Locally stored biases of the hidden layer.
  DataType hiddenBias;
  //! Locally-stored output of the preActivation function used in FreeEnergy.
  DataType preActivation;
  //! Locally stored spikeBias (hiddenSize * 1).
  DataType spikeBias;
  //! Locally stored visible Penalty (1 * 1).
  DataType visiblePenalty;
  //! Locally stored mean of the P(v|s, h).
  DataType visibleMean;
  //! Locally stored mean of the P(v|h).
  DataType spikeMean;
  //! Locally stored spike variables.
  DataType spikeSamples;
  //! Locally stored mean of the P(s|v, h).
  DataType slabMean;
  //! Locally stored slabPenalty.
  ElemType slabPenalty;
  //! Locally stored radius used for rejection sampling.
  ElemType radius;
  //! Locally-stored reconstructed output from hidden layer.
  arma::Mat<ElemType> hiddenReconstruction;
  //! Locally-stored reconstructed output from visible layer.
  arma::Mat<ElemType> visibleReconstruction;
  //! Locally-stored negative samples from Gibbs distribution.
  arma::Mat<ElemType> negativeSamples;
  //! Locally-stored gradients from the negative phase.
  arma::Mat<ElemType> negativeGradient;
  //! Locally-stored temporary negative gradient used for negative phase.
  arma::Mat<ElemType> tempNegativeGradient;
  //! Locally-stored gradient for positive phase.
  arma::Mat<ElemType> positiveGradient;
  //! Locally-stored temporary output of Gibbs chain.
  arma::Mat<ElemType> gibbsTemporary;
  //! Locally-stored persistent CD-k boolean flag.
  bool persistence;
  //! Locally-stored reset variable.
  bool reset;
};

} // namespace mlpack

#include "rbm_impl.hpp"
#include "spike_slab_rbm_impl.hpp"

#endif
