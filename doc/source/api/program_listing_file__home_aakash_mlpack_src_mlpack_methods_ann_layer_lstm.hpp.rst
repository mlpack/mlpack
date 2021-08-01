
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lstm.hpp:

Program Listing for File lstm.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_lstm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/lstm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_LSTM_HPP
   #define MLPACK_METHODS_ANN_LAYER_LSTM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <limits>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class LSTM
   {
    public:
     LSTM();
   
     LSTM(const size_t inSize,
          const size_t outSize,
          const size_t rho = std::numeric_limits<size_t>::max());
   
     LSTM(const LSTM& layer);
   
     LSTM(LSTM&&);
   
     LSTM& operator=(const LSTM& layer);
   
     LSTM& operator=(LSTM&& layer);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input,
                  OutputType& output,
                  OutputType& cellState,
                  bool useCellState = false);
   
     template<typename InputType, typename ErrorType, typename GradientType>
     void Backward(const InputType& input,
                   const ErrorType& gy,
                   GradientType& g);
   
     /*
      * Reset the layer parameter.
      */
     void Reset();
   
     /*
      * Resets the cell to accept a new input. This breaks the BPTT chain starts a
      * new one.
      *
      * @param size The current maximum number of steps through time.
      */
     void ResetCell(const size_t size);
   
     /*
      * Calculate the gradient using the output delta and the input activation.
      *
      * @param input The input parameter used for calculating the gradient.
      * @param error The calculated error.
      * @param gradient The calculated gradient.
      */
     template<typename InputType, typename ErrorType, typename GradientType>
     void Gradient(const InputType& input,
                   const ErrorType& error,
                   GradientType& gradient);
   
     size_t Rho() const { return rho; }
     size_t& Rho() { return rho; }
   
     OutputDataType const& Parameters() const { return weights; }
     OutputDataType& Parameters() { return weights; }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     OutputDataType const& Gradient() const { return grad; }
     OutputDataType& Gradient() { return grad; }
   
     size_t InSize() const { return inSize; }
   
     size_t OutSize() const { return outSize; }
   
     size_t WeightSize() const
     {
       return (4 * outSize * inSize + 7 * outSize + 4 * outSize * outSize);
     }
   
     size_t InputShape() const
     {
       return inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t inSize;
   
     size_t outSize;
   
     size_t rho;
   
     size_t forwardStep;
   
     size_t backwardStep;
   
     size_t gradientStep;
   
     OutputDataType weights;
   
     OutputDataType prevOutput;
   
     size_t batchSize;
   
     size_t batchStep;
   
     size_t gradientStepIdx;
   
     OutputDataType cellActivationError;
   
     OutputDataType delta;
   
     OutputDataType grad;
   
     OutputDataType outputParameter;
   
     OutputDataType output2GateInputWeight;
   
     OutputDataType input2GateInputWeight;
   
     OutputDataType input2GateInputBias;
   
     OutputDataType cell2GateInputWeight;
   
     OutputDataType output2GateForgetWeight;
   
     OutputDataType input2GateForgetWeight;
   
     OutputDataType input2GateForgetBias;
   
     OutputDataType cell2GateForgetWeight;
   
     OutputDataType output2GateOutputWeight;
   
     OutputDataType input2GateOutputWeight;
   
     OutputDataType input2GateOutputBias;
   
     OutputDataType cell2GateOutputWeight;
   
     OutputDataType inputGate;
   
     OutputDataType forgetGate;
   
     OutputDataType hiddenLayer;
   
     OutputDataType outputGate;
   
     OutputDataType inputGateActivation;
   
     OutputDataType forgetGateActivation;
   
     OutputDataType outputGateActivation;
   
     OutputDataType hiddenLayerActivation;
   
     OutputDataType input2HiddenWeight;
   
     OutputDataType input2HiddenBias;
   
     OutputDataType output2HiddenWeight;
   
     OutputDataType cell;
   
     OutputDataType cellActivation;
   
     OutputDataType forgetGateError;
   
     OutputDataType outputGateError;
   
     OutputDataType prevError;
   
     OutputDataType outParameter;
   
     OutputDataType inputCellError;
   
     OutputDataType inputGateError;
   
     OutputDataType hiddenError;
   
     size_t rhoSize;
   
     size_t bpttSteps;
   }; // class LSTM
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "lstm_impl.hpp"
   
   #endif
