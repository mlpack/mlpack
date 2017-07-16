/**
 * @file binary_rbm.hpp
 * @author Kris Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RBM_BINARY_LAYER_IMPL_HPP
#define MLPACK_METHODS_RBM_BINARY_LAYER_IMPL_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>

#include "binary_layer.hpp"

namespace mlpack{
namespace rbm{
class BinaryRBM
{
 public:
  // Intialise the visible and hiddenl layer of the network
  BinaryRBM(BinaryLayer<> visible, BinaryLayer<> hidden):
  visible(visible), hidden(hidden)
  {
    parameter.set_size(visible.Parameters().n_elem, 1);
  };
  // Reset function
  void Reset()
  {
    size_t weight = 0;
    weight+= visible.Parameters().n_elem;
    visible.Parameters() = arma::mat(parameter.memptr(), weight, 1, false,
        false);
    hidden.Parameters() = arma::mat(parameter.memptr(), weight, 1, false,
        false);
    visible.Reset();
    hidden.Reset();

    size_t numVisible = visible.Bias().n_rows;
    size_t numHidden = hidden.Bias().n_rows;
    // Variable for Negative grad wrt weights
    weightNegativeGrad = arma::mat(
        negativeGradient.memptr(),
        visible.Weight().n_rows,
        visible.Weight().n_cols, false, false);

    // Variable for Negative grad wrt HiddenBias
    hiddenBiasNegativeGrad = arma::mat(
        negativeGradient.memptr() + weightNegativeGrad.n_elem,
        numHidden, 1, false, false);

    // Variable for Negative grad wrt VisibleBias
    visibleBiasNegativeGrad = arma::mat(
        negativeGradient.memptr() + weightNegativeGrad.n_elem +
        hiddenBiasNegativeGrad.n_elem,
        numVisible, 1, false, false);

    // Variable for Positive grad wrt weights
    weightPositiveGrad = arma::mat(positiveGradient.memptr(),
      visible.Weight().n_rows,
      visible.Weight().n_cols, false, false);

    // Variable for Positive grad wrt hidden bias
    hiddenBiasPositiveGrad = arma::mat(
      positiveGradient.memptr() + weightPositiveGrad.n_elem,
      numHidden, 1, false, false);

    // Variable for Positive grad wrt visible bias
    visibleBiasPositiveGrad = arma::mat(
        positiveGradient.memptr() + weightPositiveGrad.n_elem +
        hiddenBiasPositiveGrad.n_elem,
        numVisible, 1, false, false);
  }

  /**
   * Free energy of the spike and slab variable
   * the free energy of the ssRBM is given my
   *
   * @param input the visible layer
   */ 
  double FreeEnergy(arma::mat&& input)
  {
    visible.ForwardPreActivation(std::move(input), std::move(preActivation));
    SoftplusFunction::Fn(preActivation, preActivation);
    return  -(arma::accu(preActivation) + arma::dot(input, visible.Bias()));
  }

  double Evaluate(arma::mat& predictors, size_t i)
  {
    size_t idx = RandInt(0, predictors.n_rows);
    arma::mat temp = arma::round(predictors.col(i));
    corruptInput.row(idx) = 1 - corruptInput.row(idx);
    return std::log(LogisticFunction::Fn(FreeEnergy(std::move(corruptInput)) -
        FreeEnergy(std::move(temp)))) * predictors.n_rows;
  }

  /**
   * Positive Gradient function. This function calculates the positive
   * phase for the binary rbm gradient calculation
   * 
   * @param input the visible layer type
   */
  void PositivePhase(arma::mat&& input)
  {
    visible.Forward(std::move(input), std::move(hiddenBiasPositiveGrad));
    weightPositiveGrad = hiddenBiasPositiveGrad * input.t();
    visibleBiasPositiveGrad = input;
  }

  /**
   * Negative Gradient function. This function calculates the negative
   * phase for the binary rbm gradient calculation
   * 
   * @param input the negative samples sampled from gibbs distribution
   */
  void NegativePhase(arma::mat&& negativeSamples)
  {
    // Collect the negative gradients
    visible.Forward(std::move(negativeSamples),
        std::move(hiddenBiasNegativeGrad));
    weightNegativeGrad = hiddenBiasNegativeGrad * negativeSamples.t();
    visibleBiasNegativeGrad = negativeSamples;
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
  BinaryLayer<>& VisibleLayer()  { return visible; }
  //! Modify the hidden layer of the network
  BinaryLayer<>& HiddenLayer()  { return hidden; }

  //! Retrun the visible layer of the network
  const BinaryLayer<>& VisibleLayer() const { return visible; }
  //! Return the hidden layer of the network
  const BinaryLayer<>& HiddenLayer() const { return hidden; }

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
  // Parameter weights of the network
  arma::mat parameter;
  // Visible layer
  BinaryLayer<> visible;
  // Hidden Layer
  BinaryLayer<> hidden;
  //! Locally-stored negative samples from gibbs Distribution
  arma::mat negativeSamples;
  //! Locally-stored gradient for  negative phase
  arma::mat negativeGradient;
  //! Locally-stored gradient for positive phase
  arma::mat positiveGradient;
  //! Locally-stored gradient wrt weight for negative phase
  arma::mat weightNegativeGrad;
  //! Locally-stored gradient wrt hidden bias for negative phase
  arma::mat hiddenBiasNegativeGrad;
  //! Locally-stored gradient wrt visible bias for negative phase
  arma::mat visibleBiasNegativeGrad;
  //! Locally-stored corrupInput used for Pseudo-Likelihood
  arma::mat corruptInput;
  //! Locally-stored gradient wrt weight for positive phase
  arma::mat weightPositiveGrad;
  //! Locally-stored gradient wrt hidden bias for positive phase
  arma::mat hiddenBiasPositiveGrad;
  //! Locally-stored gradient wrt visible bias for positive phase
  arma::mat visibleBiasPositiveGrad;

  //! Locally-stored output of the preActivation function used in FreeEnergy
  arma::mat preActivation;
  //! Locally-stored corrupInput used for Pseudo-Likelihood
};
} // namespace rbm
} // namespace mlpack
#endif
