/**
 * @file lstm_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the LSTMLayer class, which implements a lstm network
 * layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LSTM_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_LSTM_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a lstm network layer.
 *
 * This class allows specification of the type of the activation functions used
 * for the gates and cells and also of the type of the function used to
 * initialize and update the peephole weights.
 *
 * @tparam GateActivationFunction Activation function used for the gates.
 * @tparam StateActivationFunction Activation function used for the state.
 * @tparam OutputActivationFunction Activation function used for the output.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    class GateActivationFunction = LogisticFunction,
    class StateActivationFunction = TanhFunction,
    class OutputActivationFunction = TanhFunction,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class LSTMLayer
{
 public:
  /**
   * Create the LSTMLayer object using the specified parameters.
   *
   * @param outSize The number of output units.
   * @param peepholes The flag used to indicate if peephole connections should
   *        be used (Default: false).
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  LSTMLayer(const size_t outSize, const bool peepholes = false) :
      outSize(outSize),
      peepholes(peepholes),
      seqLen(1),
      offset(0)
  {
    if (peepholes)
    {
      peepholeWeights.set_size(outSize, 3);
      peepholeDerivatives = arma::zeros<OutputDataType>(outSize, 3);
    }
    else
    {
      peepholeWeights.set_size(0, 0);
    }
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    if (inGate.n_cols < seqLen)
    {
      inGate = arma::zeros<InputDataType>(outSize, seqLen);
      inGateAct = arma::zeros<InputDataType>(outSize, seqLen);
      inGateError = arma::zeros<InputDataType>(outSize, seqLen);
      outGate = arma::zeros<InputDataType>(outSize, seqLen);
      outGateAct = arma::zeros<InputDataType>(outSize, seqLen);
      outGateError = arma::zeros<InputDataType>(outSize, seqLen);
      forgetGate = arma::zeros<InputDataType>(outSize, seqLen);
      forgetGateAct = arma::zeros<InputDataType>(outSize, seqLen);
      forgetGateError = arma::zeros<InputDataType>(outSize, seqLen);
      state = arma::zeros<InputDataType>(outSize, seqLen);
      stateError = arma::zeros<InputDataType>(outSize, seqLen);
      cellAct = arma::zeros<InputDataType>(outSize, seqLen);
    }

    // Split up the inputactivation into the 3 parts (inGate, forgetGate,
    // outGate).
    inGate.col(offset) = input.submat(0, 0, outSize - 1, 0);

    forgetGate.col(offset) = input.submat(outSize, 0, (outSize * 2) - 1, 0);
    outGate.col(offset) = input.submat(outSize * 3, 0, (outSize * 4) - 1, 0);

    if (peepholes && offset > 0)
    {
      inGate.col(offset) += peepholeWeights.col(0) % state.col(offset - 1);
      forgetGate.col(offset) += peepholeWeights.col(1) %
          state.col(offset - 1);
    }

    arma::Col<eT> inGateActivation = inGateAct.unsafe_col(offset);
    GateActivationFunction::fn(inGate.unsafe_col(offset), inGateActivation);

    arma::Col<eT> forgetGateActivation = forgetGateAct.unsafe_col(offset);
    GateActivationFunction::fn(forgetGate.unsafe_col(offset),
        forgetGateActivation);

    arma::Col<eT> cellActivation = cellAct.unsafe_col(offset);
    StateActivationFunction::fn(input.submat(outSize * 2, 0,
        (outSize * 3) - 1, 0), cellActivation);

    state.col(offset) = inGateAct.col(offset) % cellActivation;

    if (offset > 0)
      state.col(offset) += forgetGateAct.col(offset) % state.col(offset - 1);

    if (peepholes)
      outGate.col(offset) += peepholeWeights.col(2) % state.col(offset);

    arma::Col<eT> outGateActivation = outGateAct.unsafe_col(offset);
    GateActivationFunction::fn(outGate.unsafe_col(offset), outGateActivation);

    OutputActivationFunction::fn(state.unsafe_col(offset), output);
    output = outGateAct.col(offset) % output;

    offset = (offset + 1) % seqLen;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename eT>
  void Backward(const InputType& /* unused */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    queryOffset = seqLen - offset - 1;

    arma::Col<eT> outGateDerivative;
    GateActivationFunction::deriv(outGateAct.unsafe_col(queryOffset),
        outGateDerivative);

    arma::Col<eT> stateActivation;
    StateActivationFunction::fn(state.unsafe_col(queryOffset), stateActivation);

    outGateError.col(queryOffset) = outGateDerivative % gy % stateActivation;

    arma::Col<eT> stateDerivative;
    StateActivationFunction::deriv(stateActivation, stateDerivative);

    stateError.col(queryOffset) = gy % outGateAct.col(queryOffset) %
        stateDerivative;

    if (queryOffset < (seqLen - 1))
    {
      stateError.col(queryOffset) += stateError.col(queryOffset + 1) %
          forgetGateAct.col(queryOffset + 1);

      if (peepholes)
      {
        stateError.col(queryOffset) += inGateError.col(queryOffset + 1) %
            peepholeWeights.col(0);
        stateError.col(queryOffset) += forgetGateError.col(queryOffset + 1) %
            peepholeWeights.col(1);
      }
    }

    if (peepholes)
    {
      stateError.col(queryOffset) += outGateError.col(queryOffset) %
          peepholeWeights.col(2);
    }

    arma::Col<eT> cellDerivative;
    StateActivationFunction::deriv(cellAct.col(queryOffset), cellDerivative);

    arma::Col<eT> cellError = inGateAct.col(queryOffset) % cellDerivative %
        stateError.col(queryOffset);

    if (queryOffset > 0)
    {
      arma::Col<eT> forgetGateDerivative;
      GateActivationFunction::deriv(forgetGateAct.col(queryOffset),
          forgetGateDerivative);

      forgetGateError.col(queryOffset) = forgetGateDerivative %
          stateError.col(queryOffset) % state.col(queryOffset - 1);
    }

    arma::Col<eT> inGateDerivative;
    GateActivationFunction::deriv(inGateAct.col(queryOffset), inGateDerivative);

    inGateError.col(queryOffset) = inGateDerivative %
        stateError.col(queryOffset) % cellAct.col(queryOffset);

    if (peepholes)
    {
      peepholeDerivatives.col(2) += outGateError.col(queryOffset) %
          state.col(queryOffset);

      if (queryOffset > 0)
      {
        peepholeDerivatives.col(0) += inGateError.col(queryOffset) %
            state.col(queryOffset - 1);
        peepholeDerivatives.col(1) += forgetGateError.col(queryOffset) %
            state.col(queryOffset - 1);
      }
    }

    g = arma::zeros<arma::Mat<eT> >(outSize * 4, 1);
    g.submat(0, 0, outSize - 1, 0) = inGateError.col(queryOffset);
    g.submat(outSize, 0, (outSize * 2) - 1, 0) =
        forgetGateError.col(queryOffset);
    g.submat(outSize * 2, 0, (outSize * 3) - 1, 0) = cellError;
    g.submat(outSize * 3, 0, (outSize * 4) - 1, 0) =
        outGateError.col(queryOffset);

    offset = (offset + 1) % seqLen;
  }

  /**
   * Ordinary feed backward pass of the lstm layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename InputType, typename eT, typename GradientDataType>
  void Gradient(const InputType& /* input */,
                const arma::Mat<eT>& /* gy */,
                GradientDataType& /* g */)
  {
    if (peepholes && offset == 0)
    {
      peepholeGradient.col(0) = arma::trans((peepholeWeights.col(0).t() *
          (inGateError.col(queryOffset) % peepholeDerivatives.col(0))) *
          inGate.col(queryOffset).t());

      peepholeGradient.col(1) = arma::trans((peepholeWeights.col(1).t() *
          (forgetGateError.col(queryOffset) % peepholeDerivatives.col(1))) *
          forgetGate.col(queryOffset).t());

      peepholeGradient.col(2) = arma::trans((peepholeWeights.col(2).t() *
          (outGateError.col(queryOffset) % peepholeDerivatives.col(2))) *
          outGate.col(queryOffset).t());

      peepholeDerivatives.zeros();
    }
  }

  //! Get the peephole weights.
  OutputDataType const& Weights() const { return peepholeWeights; }
  //! Modify the peephole weights.
  OutputDataType& Weights() { return peepholeWeights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the peephole gradient.
  OutputDataType const& Gradient() const { return peepholeGradient; }
  //! Modify the peephole gradient.
  OutputDataType& Gradient() { return peepholeGradient; }

  //! Get the sequence length.
  size_t SeqLen() const { return seqLen; }
  //! Modify the sequence length.
  size_t& SeqLen() { return seqLen; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(peepholes, "peepholes");

    if (peepholes)
    {
      ar & data::CreateNVP(peepholeWeights, "peepholeWeights");

      if (Archive::is_loading::value)
      {
        peepholeDerivatives = arma::zeros<OutputDataType>(
            peepholeWeights.n_rows, 3);
      }
    }
  }

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored peephole indication flag.
  bool peepholes;

  //! Locally-stored length of the the input sequence.
  size_t seqLen;

  //! Locally-stored sequence offset.
  size_t offset;

  //! Locally-stored query offset.
  size_t queryOffset;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored ingate object.
  InputDataType inGate;

  //! Locally-stored ingate activation object.
  InputDataType inGateAct;

  //! Locally-stored ingate error object.
  InputDataType inGateError;

  //! Locally-stored outgate object.
  InputDataType outGate;

  //! Locally-stored outgate activation object.
  InputDataType outGateAct;

  //! Locally-stored outgate error object.
  InputDataType outGateError;

  //! Locally-stored forget object.
  InputDataType forgetGate;

  //! Locally-stored forget activation object.
  InputDataType forgetGateAct;

  //! Locally-stored forget error object.
  InputDataType forgetGateError;

  //! Locally-stored state object.
  InputDataType state;

  //! Locally-stored state erro object.
  InputDataType stateError;

  //! Locally-stored cell activation object.
  InputDataType cellAct;

  //! Locally-stored peephole weight object.
  OutputDataType peepholeWeights;

  //! Locally-stored derivatives object.
  OutputDataType peepholeDerivatives;

  //! Locally-stored peephole gradient object.
  OutputDataType peepholeGradient;
}; // class LSTMLayer

//! Layer traits for the lstm layer.
template<
    class GateActivationFunction,
    class StateActivationFunction,
    class OutputActivationFunction,
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<LSTMLayer<GateActivationFunction,
                            StateActivationFunction,
                            OutputActivationFunction,
                            InputDataType,
                            OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = true;
  static const bool IsConnection = false;
};

} // namespace ann
} // namespace mlpack

#endif
