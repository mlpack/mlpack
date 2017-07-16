/**
 * @file ssRBM.hpp
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RBM_SSRBM_HPP
#define MLPACK_METHODS_RBM_SSRBM_HPP
// In case it hasn't yet been included.
#include "spike_slab_layer.hpp"

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>

namespace mlpack{
namespace rbm{
class ssRBM
{
 public:
  // Intialise the visible and hiddenl layer of the network
  ssRBM(SpikeSlabLayer<> visible, SpikeSlabLayer<> hidden):
  visible(visible), hidden(hidden)
  {
    parameter.set_size(visible.Parameters().n_elem, 1);
    positiveGradient.set_size(visible.Parameters().n_elem, 1);
    negativeGradient.set_size(visible.Parameters().n_elem, 1);
  };
  // Reset function
  void Reset()
  {
  size_t weight = 0;
  numVisible = visible.InSize();
  numHidden = visible.OutSize();
  weight+= visible.Parameters().n_elem;
  visible.Parameters() = arma::mat(parameter.memptr(), weight, 1, false,
      false);
  hidden.Parameters() = arma::mat(parameter.memptr(), weight, 1, false,
      false);
  // Reset the parameters of all the layers and Gradients
  visible.Reset();
  hidden.Reset();
  ResetGradient();

  // Variable for Negative grad wrt weights
  weightNegativeGrad = arma::cube(
      negativeGradient.memptr(),
      visible.PoolSize(),
      visible.InSize(),
      visible.OutSize(),
      false, false);

  // Variable for Negative grad wrt HiddenBias(Spike Bias)
  hiddenBiasNegativeGrad = arma::mat(
      negativeGradient.memptr() + weightNegativeGrad.n_elem,
      1, numHidden, false, false);

  // Variable for Negative grad wrt VisibleBias(Lambda Bias)
  visibleBiasNegativeGrad = arma::mat(
      negativeGradient.memptr() + weightNegativeGrad.n_elem +
      hiddenBiasNegativeGrad.n_elem,
      numVisible, 1, false, false);

  // Variable for Positive grad wrt weights
  weightPositiveGrad = arma::cube(positiveGradient.memptr(),
    visible.PoolSize(),
      visible.InSize(),
      visible.OutSize(),
    false, false);

  // Variable for Positive grad wrt hidden bias
  hiddenBiasPositiveGrad = arma::mat(
    positiveGradient.memptr() + weightPositiveGrad.n_elem,
    1, numHidden, false, false);

  // Variable for Positive grad wrt visible bias
  visibleBiasPositiveGrad = arma::mat(
      positiveGradient.memptr() + weightPositiveGrad.n_elem +
      hiddenBiasPositiveGrad.n_elem,
      numVisible, 1, false, false);
  }

  void ResetGradient()
  {
    weightPositiveGrad.set_size(visible.Weight().n_rows,
        visible.Weight().n_cols,
        visible.Weight().n_slices);

    weightNegativeGrad.set_size(visible.Weight().n_rows,
        visible.Weight().n_cols,
        visible.Weight().n_slices);

    visibleBiasPositiveGrad.set_size(numVisible, 1);
    visibleBiasNegativeGrad.set_size(numVisible, 1);

    hiddenBiasPositiveGrad.set_size(1, numHidden);
    hiddenBiasNegativeGrad.set_size(1, numHidden);
  }

  /**
   * Free energy of the spike and slab variable
   * the free energy of the ssRBM is given my
   * $v^t$$\Delta$v - $\sum_{i=1}^N$ 
   * $\log{ \sqrt{\frac{(-2\pi)^K}{\prod_{m=1}^{K}(\alpha_i)_m}}}$ -
   * $\sum_{i=1}^N \log(1+\exp( b_i +
   * \sum_{m=1}^k \frac{(v(w_i)_m^t)^2}{2(\alpha_i)_m})$
   *
   * @param input the visible layer
   */ 
  double FreeEnergy(arma::mat&& input)
  {
    const double pi = boost::math::constants::pi<double>();
    freeEnergySum = 0;
    double temp = arma::as_scalar(input.t() *
        arma::diagmat(visible.LambdaBias()) * input);
    for (size_t i = 0; i < numHidden; i++)
    {
      temp = std::log(std::sqrt(std::pow(-2 * pi, 2) /
          arma::as_scalar(arma::prod(hidden.SlabBias().col(i)))));
    }

    for (size_t i = 0; i < numHidden; i++)
    {
      for (size_t j = 0; j < hidden.SlabBias().n_cols; j++)
        freeEnergySum +=
            SoftplusFunction::Fn(arma::as_scalar(hidden.SpikeBias().col(i)) +
            std::pow(arma::as_scalar(visible.Weight().slice(i).row(j) *
                input), 2) / arma::as_scalar(hidden.SlabBias().col(i).row(j)));
    }

    return temp + freeEnergySum;
  }

  double Evaluate(arma::mat& predictors, size_t i)
  {
    return 0;
  }

  /**
   * Gradient function calculates the gradient for the spike and
   * slab RBM.
   *
   * @param input the visible input
   * @param output the computed gradient
   */
  void PositivePhase(arma::mat&& input)
  {
    positiveGradient.zeros();
    // positive phase
    visible.Sample(std::move(input), std::move(tempHidden));

    visible.Psgivenvh(std::move(input), std::move(tempHidden),
        std::move(tempMean), std::move(tempSlab));

    // positive weight gradient
    for (size_t i = 0 ; i < tempHidden.n_elem; i++)
    {
      weightPositiveGrad.slice(i) = tempMean * input.t() *
          tempHidden[i];
    }

    // positive hidden bias gradient
    for (size_t i = 0; i < visible.Weight().n_slices; i++)
      hiddenBiasPositiveGrad.col(i) = -((
          visible.Weight().slice(i) *input).t() *
          tempSlab.col(i)) +
          arma::as_scalar(visible.SpikeBias().col(i));
    // positive lambda bias
    visibleBiasPositiveGrad = (0.5 * input * input.t());
    visibleBiasPositiveGrad = visibleBiasPositiveGrad.diag();
  }

  void NegativePhase(arma::mat&& negativeSamples)
  {
    negativeGradient.zeros();
    weightNegativeGrad.zeros();

    visible.Sample(std::move(negativeSamples), std::move(tempHidden));
    visible.Psgivenvh(std::move(negativeSamples),
        std::move(tempHidden),
        std::move(tempMean), std::move(tempSlab));

    // negative weight gradient
    for (size_t j = 0; j < tempHidden.n_rows; j++)
      weightNegativeGrad.slice(j) +=  tempMean * negativeSamples.t() *
          tempHidden[j];

    // negative hidden bias gradient
    for (size_t j = 0; j < visible.Weight().n_slices; j++)
      hiddenBiasNegativeGrad.col(j) += -arma::as_scalar(
          (visible.Weight().slice(j) * negativeSamples).t() *
          tempSlab.col(j) +
          visible.SpikeBias().col(j));

    // negative lambda bias gradient
    arma::mat visibleBiasNegativeGradTemp = 0.5 *
        negativeSamples * negativeSamples.t();
    visibleBiasNegativeGrad += visibleBiasNegativeGradTemp.diag();
  }

  void SampleHidden(arma::mat&& input, arma::mat&& output)
  {
    visible.Sample(std::move(input), std::move(output));
  }

  void SampleVisible(arma::mat&& input, arma::mat&& output)
  {
    hidden.Sample(std::move(input), std::move(output));
  }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(visible, "visible");
    ar & data::CreateNVP(hidden, "hidden");
  }

  //! Modify the visible layer of the network
  SpikeSlabLayer<>& VisibleLayer()  { return visible; }
  //! Modify the hidden layer of the network
  SpikeSlabLayer<>& HiddenLayer()  { return hidden; }

  //! Retrun the visible layer of the network
  const SpikeSlabLayer<>& VisibleLayer() const { return visible; }
  //! Return the hidden layer of the network
  const SpikeSlabLayer<>& HiddenLayer() const { return hidden; }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameter; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameter; }

  //! Return the initial point for the optimization.
  const arma::mat& PositiveGradient() const { return positiveGradient; }
  //! Modify the initial point for the optimization.
  arma::mat& PositiveGradient() { return positiveGradient; }

  //! Return the initial point for the optimization.
  const arma::mat& NegativeGradient() const { return negativeGradient; }
  //! Modify the initial point for the optimization.
  arma::mat& NegativeGradient() { return negativeGradient; }


 private:
  //! Locally stored parameters
  arma::mat parameter;
  //! Locally stored parameters number of hidden neurons
  size_t numHidden;
  //! Locally stored parameters number of visible neurons
  size_t numVisible;
  //! Locally stored visible layer
  SpikeSlabLayer<> visible;
  //! Locally stored hidden layer
  SpikeSlabLayer<> hidden;
  //! Locally stored parmeter used in freeEnergy calculation
  double freeEnergySum;
  //! Locally stored temp Hidden layer value used in free energy
  arma::vec tempHidden;
  //! Locally stored temp Mean valued for P(s|v,h) used in free energy
  arma::mat tempMean;
  //! Locally stored temp Slab valued for P(s|v,h) used in free energy
  arma::mat tempSlab;
  //! Locally-stored negative samples from gibbs Distribution
  arma::mat negativeSamples;
  //! Locally-stored gradient for  negative phase
  arma::mat negativeGradient;
  //! Locally-stored gradient for positive phase
  arma::mat positiveGradient;
  //! Locally-stored gradient wrt weight for negative phase
  arma::cube weightNegativeGrad;
  //! Locally-stored gradient wrt hidden bias for negative phase
  arma::mat hiddenBiasNegativeGrad;
  //! Locally-stored gradient wrt visible bias for negative phase
  arma::mat visibleBiasNegativeGrad;

  //! Locally-stored gradient wrt weight for positive phase
  arma::cube weightPositiveGrad;
  //! Locally-stored gradient wrt hidden bias for positive phase
  arma::mat hiddenBiasPositiveGrad;
  //! Locally-stored gradient wrt visible bias for positive phase
  arma::mat visibleBiasPositiveGrad;
};
} // namespace rbm
} // namespace mlpack
#endif
