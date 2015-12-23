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
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a lstm network layer.
 *
 * This class allows specification of the type of the activation functions used
 * for the gates and cells and also of the type of the function used to
 * initialize and update the peephole weights.
 *
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam GateActivationFunction Activation function used for the gates.
 * @tparam StateActivationFunction Activation function used for the state.
 * @tparam OutputActivationFunction Activation function used for the output.
 * @tparam WeightInitRule Rule used to initialize the weight matrix.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam PeepholeDataType Type of the peephole data (weights, derivatives and
 *         gradients).
 */
template <
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    class GateActivationFunction = LogisticFunction,
    class StateActivationFunction = TanhFunction,
    class OutputActivationFunction = TanhFunction,
    class WeightInitRule = RandomInitialization,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    typename PeepholeDataType = arma::cube
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
  LSTMLayer(const size_t outSize,
            const bool peepholes = false,
            WeightInitRule weightInitRule = WeightInitRule()) :
      outSize(outSize),
      peepholes(peepholes),
      seqLen(1),
      offset(0),
      optimizer(new OptimizerType<LSTMLayer<OptimizerType,
                                            GateActivationFunction,
                                            StateActivationFunction,
                                            OutputActivationFunction,
                                            WeightInitRule,
                                            InputDataType,
                                            OutputDataType,
                                            PeepholeDataType>,
                                            PeepholeDataType>(*this)),
      ownsOptimizer(true)
  {
    if (peepholes)
    {
      weightInitRule.Initialize(peepholeWeights, outSize, 1, 3);
      peepholeDerivatives = PeepholeDataType(outSize, 1, 3);
      peepholeGradient = PeepholeDataType(outSize, 1, 3);
    }
  }

  /**
   * Delete the LSTMLayer object and its optimizer.
   */
  ~LSTMLayer()
  {
    if (ownsOptimizer)
      delete optimizer;
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
      inGate.col(offset) += peepholeWeights.slice(0) % state.col(offset - 1);
      forgetGate.col(offset) += peepholeWeights.slice(1) %
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
      outGate.col(offset) += peepholeWeights.slice(2) % state.col(offset);

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
    size_t queryOffset = seqLen - offset - 1;

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
            peepholeWeights.slice(0);
        stateError.col(queryOffset) += forgetGateError.col(queryOffset + 1) %
            peepholeWeights.slice(1);
      }
    }

    if (peepholes)
    {
      stateError.col(queryOffset) += outGateError.col(queryOffset) %
          peepholeWeights.slice(2);
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
      peepholeDerivatives.slice(2) += outGateError.col(queryOffset) %
          state.col(queryOffset);

      if (queryOffset > 0)
      {
        peepholeDerivatives.slice(0) += inGateError.col(queryOffset) %
            state.col(queryOffset - 1);
        peepholeDerivatives.slice(1) += forgetGateError.col(queryOffset) %
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

    if (peepholes && offset == 0)
    {
      peepholeGradient.slice(0) = arma::trans((peepholeWeights.slice(0).t() *
          (inGateError.col(queryOffset) % peepholeDerivatives.slice(0))) *
          inGate.col(queryOffset).t());

      peepholeGradient.slice(1) = arma::trans((peepholeWeights.slice(1).t() *
          (forgetGateError.col(queryOffset) % peepholeDerivatives.slice(1))) *
          forgetGate.col(queryOffset).t());

      peepholeGradient.slice(2) = arma::trans((peepholeWeights.slice(2).t() *
          (outGateError.col(queryOffset) % peepholeDerivatives.slice(2))) *
          outGate.col(queryOffset).t());

      optimizer->Update();
      optimizer->Optimize();
      optimizer->Reset();
      peepholeDerivatives.zeros();
    }
  }

  //! Get the peephole weights.
  PeepholeDataType& Weights() const { return peepholeWeights; }
  //! Modify the peephole weights.
  PeepholeDataType& Weights() { return peepholeWeights; }

  //! Get the input parameter.
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the peephole gradient.
  PeepholeDataType& Gradient() const {return peepholeGradient; }
  //! Modify the peephole gradient.
  PeepholeDataType& Gradient() { return peepholeGradient; }

  //! Get the sequence length.
  size_t SeqLen() const { return seqLen; }
  //! Modify the sequence length.
  size_t& SeqLen() { return seqLen; }

 private:
  //! Locally-stored number of output units.
  const size_t outSize;

  //! Locally-stored peephole indication flag.
  const bool peepholes;

  //! Locally-stored length of the the input sequence.
  size_t seqLen;

  //! Locally-stored sequence offset.
  size_t offset;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<LSTMLayer<OptimizerType,
                          GateActivationFunction,
                          StateActivationFunction,
                          OutputActivationFunction,
                          WeightInitRule,
                          InputDataType,
                          OutputDataType,
                          PeepholeDataType>, PeepholeDataType>* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;

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
  PeepholeDataType peepholeWeights;

  //! Locally-stored derivatives object.
  PeepholeDataType peepholeDerivatives;

  //! Locally-stored peephole gradient object.
  PeepholeDataType peepholeGradient;
}; // class LSTMLayer

//! Layer traits for the lstm layer.
template<
    template<typename, typename> class OptimizerType,
    class GateActivationFunction,
    class StateActivationFunction,
    class OutputActivationFunction,
    class WeightInitRule,
    typename InputDataType,
    typename OutputDataType,
    typename PeepholeDataType
>
class LayerTraits<LSTMLayer<OptimizerType,
                            GateActivationFunction,
                            StateActivationFunction,
                            OutputActivationFunction,
                            WeightInitRule,
                            InputDataType,
                            OutputDataType,
                            PeepholeDataType> >
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
