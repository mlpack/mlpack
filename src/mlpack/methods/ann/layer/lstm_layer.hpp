/**
 * @file lstm_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the LSTMLayer class, which implements a lstm network
 * layer.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_LSTM_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_LSTM_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/optimizer/steepest_descent.hpp>

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
 * @tparam WeightInitRule Rule used to initialize the weight matrix.
 * @tparam MatType Type of data (arma::mat or arma::sp_mat).
 * @tparam VecType Type of data (arma::colvec, arma::mat or arma::sp_mat).
 */
template <
    class GateActivationFunction = LogisticFunction,
    class StateActivationFunction = TanhFunction,
    class OutputActivationFunction = TanhFunction,
    class WeightInitRule = NguyenWidrowInitialization,
    typename OptimizerType = SteepestDescent<>,
    typename MatType = arma::mat,
    typename VecType = arma::colvec
>
class LSTMLayer
{
 public:
  /**
   * Create the LSTMLayer object using the specified parameters.
   *
   * @param layerSize The number of memory cells.
   * @param layerSize The length of the input sequence.
   * @param peepholes The flag used to indicate if peephole connections should
   * be used (Default: true).
   * @param WeightInitRule The weight initialize rule used to initialize the
   * peephole connection matrix.
   */
  LSTMLayer(const size_t layerSize,
            const size_t seqLen = 1,
            const bool peepholes = false,
            WeightInitRule weightInitRule = WeightInitRule()) :
      inputActivations(arma::zeros<VecType>(layerSize * 4)),
      layerSize(layerSize),
      seqLen(seqLen),
      inGate(arma::zeros<MatType>(layerSize, seqLen)),
      inGateAct(arma::zeros<MatType>(layerSize, seqLen)),
      inGateError(arma::zeros<MatType>(layerSize, seqLen)),
      outGate(arma::zeros<MatType>(layerSize, seqLen)),
      outGateAct(arma::zeros<MatType>(layerSize, seqLen)),
      outGateError(arma::zeros<MatType>(layerSize, seqLen)),
      forgetGate(arma::zeros<MatType>(layerSize, seqLen)),
      forgetGateAct(arma::zeros<MatType>(layerSize, seqLen)),
      forgetGateError(arma::zeros<MatType>(layerSize, seqLen)),
      state(arma::zeros<MatType>(layerSize, seqLen)),
      stateError(arma::zeros<MatType>(layerSize, seqLen)),
      cellAct(arma::zeros<MatType>(layerSize, seqLen)),
      offset(0),
      peepholes(peepholes)
  {
    if (peepholes)
    {
      weightInitRule.Initialize(inGatePeepholeWeights, layerSize, 1);
      inGatePeepholeDerivatives = arma::zeros<VecType>(layerSize);
      inGatePeepholeOptimizer = std::unique_ptr<OptimizerType>(
      new OptimizerType(1, layerSize));

      weightInitRule.Initialize(forgetGatePeepholeWeights, layerSize, 1);
      forgetGatePeepholeDerivatives = arma::zeros<VecType>(layerSize);
      forgetGatePeepholeOptimizer = std::unique_ptr<OptimizerType>(
      new OptimizerType(1, layerSize));

      weightInitRule.Initialize(outGatePeepholeWeights, layerSize, 1);
      outGatePeepholeDerivatives = arma::zeros<VecType>(layerSize);
      outGatePeepholeOptimizer = std::unique_ptr<OptimizerType>(
      new OptimizerType(1, layerSize));
    }
  }

  ~LSTMLayer()
  {
    OptimizerType* inGatePeepholePtr = inGatePeepholeOptimizer.release();
    delete inGatePeepholePtr;

    OptimizerType* forgetGatePeepholePtr = forgetGatePeepholeOptimizer.release();
    delete forgetGatePeepholePtr;

    OptimizerType* outGatePeepholePtr = outGatePeepholeOptimizer.release();
    delete outGatePeepholePtr;
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param inputActivation Input data used for evaluating the specified
   * activity function.
   * @param outputActivation Datatype to store the resulting output activation.
   */
  void FeedForward(const VecType& inputActivation, VecType& outputActivation)
  {
    if (inGate.n_cols < seqLen)
    {
      inGate = arma::zeros<MatType>(layerSize, seqLen);
      inGateAct = arma::zeros<MatType>(layerSize, seqLen);
      inGateError = arma::zeros<MatType>(layerSize, seqLen);
      outGate = arma::zeros<MatType>(layerSize, seqLen);
      outGateAct = arma::zeros<MatType>(layerSize, seqLen);
      outGateError = arma::zeros<MatType>(layerSize, seqLen);
      forgetGate = arma::zeros<MatType>(layerSize, seqLen);
      forgetGateAct = arma::zeros<MatType>(layerSize, seqLen);
      forgetGateError = arma::zeros<MatType>(layerSize, seqLen);
      state = arma::zeros<MatType>(layerSize, seqLen);
      stateError = arma::zeros<MatType>(layerSize, seqLen);
      cellAct = arma::zeros<MatType>(layerSize, seqLen);
    }

    // Split up the inputactivation into the 3 parts (inGate, forgetGate,
    // outGate).
    inGate.col(offset) = inputActivation.subvec(0, layerSize - 1);
    forgetGate.col(offset) = inputActivation.subvec(
        layerSize, (layerSize * 2) - 1);
    outGate.col(offset) = inputActivation.subvec(
        layerSize * 3, (layerSize * 4) - 1);

    if (peepholes && offset > 0)
    {
      inGate.col(offset) += inGatePeepholeWeights % state.col(offset - 1);
      forgetGate.col(offset) += forgetGatePeepholeWeights %
          state.col(offset - 1);
    }

    VecType inGateActivation = inGateAct.unsafe_col(offset);
    GateActivationFunction::fn(inGate.unsafe_col(offset), inGateActivation);

    VecType forgetGateActivation = forgetGateAct.unsafe_col(offset);
    GateActivationFunction::fn(forgetGate.unsafe_col(offset),
        forgetGateActivation);

    VecType cellActivation = cellAct.unsafe_col(offset);
    StateActivationFunction::fn(inputActivation.subvec(layerSize * 2,
        (layerSize * 3) - 1), cellActivation);

    state.col(offset) = inGateAct.col(offset) % cellActivation;

    if (offset > 0)
      state.col(offset) += forgetGateAct.col(offset) % state.col(offset - 1);

    if (peepholes)
      outGate.col(offset) += outGatePeepholeWeights % state.col(offset);

    VecType outGateActivation = outGateAct.unsafe_col(offset);
    GateActivationFunction::fn(outGate.unsafe_col(offset), outGateActivation);

    OutputActivationFunction::fn(state.unsafe_col(offset), outputActivation);
    outputActivation = outGateAct.col(offset) % outputActivation;

    offset = (offset + 1) % seqLen;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param inputActivation Input data used for calculating the function f(x).
   * @param error The backpropagated error.
   * @param delta The calculating delta using the partial derivative of the
   * error with respect to a weight.
   */
  void FeedBackward(const VecType& /* unused */,
                    const VecType& error,
                    VecType& delta)
  {
    size_t queryOffset = seqLen - offset - 1;

    VecType outGateDerivative;
    GateActivationFunction::deriv(outGateAct.unsafe_col(queryOffset),
        outGateDerivative);

    VecType stateActivation;
    StateActivationFunction::fn(state.unsafe_col(queryOffset), stateActivation);

    outGateError.col(queryOffset) = outGateDerivative % error % stateActivation;

    VecType stateDerivative;
    StateActivationFunction::deriv(stateActivation, stateDerivative);

    stateError.col(queryOffset) = error % outGateAct.col(queryOffset) %
        stateDerivative;

    if (queryOffset < (seqLen - 1))
    {
      stateError.col(queryOffset) += stateError.col(queryOffset + 1) %
          forgetGateAct.col(queryOffset + 1);

      if (peepholes)
      {
        stateError.col(queryOffset) += inGateError.col(queryOffset + 1) %
            inGatePeepholeWeights;
        stateError.col(queryOffset) += forgetGateError.col(queryOffset + 1) %
            forgetGatePeepholeWeights;
      }
    }

    if (peepholes)
    {
      stateError.col(queryOffset) += outGateError.col(queryOffset) %
          outGatePeepholeWeights;
    }

    VecType cellDerivative;
    StateActivationFunction::deriv(cellAct.col(queryOffset), cellDerivative);

    VecType cellError = inGateAct.col(queryOffset) % cellDerivative %
        stateError.col(queryOffset);

    if (queryOffset > 0)
    {
      VecType forgetGateDerivative;
      GateActivationFunction::deriv(forgetGateAct.col(queryOffset),
          forgetGateDerivative);

      forgetGateError.col(queryOffset) = forgetGateDerivative %
          stateError.col(queryOffset) % state.col(queryOffset - 1);
    }

    VecType inGateDerivative;
    GateActivationFunction::deriv(inGateAct.col(queryOffset), inGateDerivative);

    inGateError.col(queryOffset) = inGateDerivative %
        stateError.col(queryOffset) % cellAct.col(queryOffset);

    if (peepholes)
    {
      outGateDerivative += outGateError.col(queryOffset) %
          state.col(queryOffset);
      if (queryOffset > 0)
      {
        inGatePeepholeDerivatives += inGateError.col(queryOffset) %
            state.col(queryOffset - 1);
        forgetGatePeepholeDerivatives += forgetGateError.col(queryOffset) %
            state.col(queryOffset - 1);
      }
    }

    delta = arma::zeros<VecType>(layerSize * 4);
    delta.subvec(0, layerSize - 1) = inGateError.col(queryOffset);
    delta.subvec(layerSize, (layerSize * 2) - 1) =
        forgetGateError.col(queryOffset);
    delta.subvec(layerSize * 2, (layerSize * 3) - 1) = cellError;
    delta.subvec(layerSize * 3, (layerSize * 4) - 1) =
        outGateError.col(queryOffset);

    offset = (offset + 1) % seqLen;

    if (peepholes && offset == 0)
    {
      inGatePeepholeGradient = (inGatePeepholeWeights.t() *
          (inGateError.col(queryOffset) % inGatePeepholeDerivatives)) *
          inGate.col(queryOffset).t();

      forgetGatePeepholeGradient = (forgetGatePeepholeWeights.t() *
          (forgetGateError.col(queryOffset) % forgetGatePeepholeDerivatives)) *
          forgetGate.col(queryOffset).t();

      outGatePeepholeGradient = (outGatePeepholeWeights.t() *
          (outGateError.col(queryOffset) % outGatePeepholeDerivatives)) *
          outGate.col(queryOffset).t();

      inGatePeepholeOptimizer->UpdateWeights(inGatePeepholeWeights,
          inGatePeepholeGradient.t(), 0);

      forgetGatePeepholeOptimizer->UpdateWeights(forgetGatePeepholeWeights,
          forgetGatePeepholeGradient.t(), 0);

      outGatePeepholeOptimizer->UpdateWeights(outGatePeepholeWeights,
          outGatePeepholeGradient.t(), 0);

      inGatePeepholeDerivatives.zeros();
      forgetGatePeepholeDerivatives.zeros();
      outGatePeepholeDerivatives.zeros();
    }
  }

  //! Get the input activations.
  const VecType& InputActivation() const { return inputActivations; }
  //! Modify the input activations.
  VecType& InputActivation() { return inputActivations; }

  //! Get input size.
  size_t InputSize() const { return layerSize * 4; }

  //! Get output size.
  size_t OutputSize() const { return layerSize; }
  //! Modify the output size.
  size_t& OutputSize() { return layerSize; }

  //! Get the number of output maps.
  size_t OutputMaps() const { return 1; }

  //! Get the number of layer slices.
  size_t LayerSlices() const { return 1; }

  //! Get the number of layer rows.
  size_t LayerRows() const { return layerSize; }

  //! Get the number of layer columns.
  size_t LayerCols() const { return 1; }

  //! Get the detla.
  VecType& Delta() const { return delta; }
  //! Modify the delta.
  VecType& Delta() { return delta; }

  //! Get the sequence length.
  size_t SeqLen() const { return seqLen; }
  //! Modify the sequence length.
  size_t& SeqLen() { return seqLen; }

  //! Get the InGate peephole weights..
  MatType& InGatePeepholeWeights() const { return inGatePeepholeWeights; }
  //! Modify the InGate peephole weights..
  MatType& InGatePeepholeWeights() { return inGatePeepholeWeights; }

  //! Get the InGate peephole weights..
  MatType& ForgetGatePeepholeWeights() const {
    return forgetGatePeepholeWeights; }
  //! Modify the InGate peephole weights..
  MatType& ForgetGatePeepholeWeights() { return forgetGatePeepholeWeights; }

  //! Get the InGate peephole weights..
  MatType& OutGatePeepholeWeights() const { return outGatePeepholeWeights; }
  //! Modify the InGate peephole weights..
  MatType& OutGatePeepholeWeights() { return outGatePeepholeWeights; }

 private:
  //! Locally-stored input activation object.
  VecType inputActivations;

  //! Locally-stored delta object.
  VecType delta;

  //! Locally-stored number of memory cells.
  size_t layerSize;

  //! Locally-stored length of the the input sequence.
  size_t seqLen;

  //! Locally-stored ingate object.
  MatType inGate;

  //! Locally-stored ingate activation object.
  MatType inGateAct;

  //! Locally-stored ingate error object.
  MatType inGateError;

  //! Locally-stored outgate object.
  MatType outGate;

  //! Locally-stored outgate activation object.
  MatType outGateAct;

  //! Locally-stored outgate error object.
  MatType outGateError;

  //! Locally-stored forget object.
  MatType forgetGate;

  //! Locally-stored forget activation object.
  MatType forgetGateAct;

  //! Locally-stored forget error object.
  MatType forgetGateError;

  //! Locally-stored state object.
  MatType state;

  //! Locally-stored state erro object.
  MatType stateError;

  //! Locally-stored cell activation object.
  MatType cellAct;

  //! Locally-stored sequence offset.
  size_t offset;

  //! Locally-stored peephole indication flag.
  const bool peepholes;

  //! Locally-stored peephole ingate weights.
  MatType inGatePeepholeWeights;

  //! Locally-stored peephole ingate derivatives.
  VecType inGatePeepholeDerivatives;

  //! Locally-stored peephole ingate gradients.
  MatType inGatePeepholeGradient;

  //! Locally-stored ingate peephole optimzer object.
  std::unique_ptr<OptimizerType> inGatePeepholeOptimizer;

  //! Locally-stored peephole forget weights.
  MatType forgetGatePeepholeWeights;

  //! Locally-stored peephole forget derivatives.
  VecType forgetGatePeepholeDerivatives;

  //! Locally-stored peephole forget gradients.
  MatType forgetGatePeepholeGradient;

  //! Locally-stored forget peephole optimzer object.
  std::unique_ptr<OptimizerType> forgetGatePeepholeOptimizer;

  //! Locally-stored peephole outgate weights.
  MatType outGatePeepholeWeights;

  //! Locally-stored peephole outgate derivatives.
  VecType outGatePeepholeDerivatives;

  //! Locally-stored peephole outgate gradients.
  MatType outGatePeepholeGradient;

  //! Locally-stored outgate peephole optimzer object.
  std::unique_ptr<OptimizerType> outGatePeepholeOptimizer;
}; // class LSTMLayer

//! Layer traits for the bias layer.
template<
    class GateActivationFunction,
    class StateActivationFunction,
    class OutputActivationFunction,
    class WeightInitRule,
    typename OptimizerType,
    typename MatType,
    typename VecType
>
class LayerTraits<
    LSTMLayer<GateActivationFunction,
    StateActivationFunction,
    OutputActivationFunction,
    WeightInitRule,
    OptimizerType,
    MatType,
    VecType>
>
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = true;
};

}; // namespace ann
}; // namespace mlpack

#endif
