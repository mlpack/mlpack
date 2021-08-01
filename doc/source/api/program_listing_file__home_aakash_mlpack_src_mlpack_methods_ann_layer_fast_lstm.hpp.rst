
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_fast_lstm.hpp:

Program Listing for File fast_lstm.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_fast_lstm.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/fast_lstm.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_FAST_LSTM_HPP
   #define MLPACK_METHODS_ANN_LAYER_FAST_LSTM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <limits>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class FastLSTM
   {
    public:
     // Convenience typedefs.
     typedef typename InputDataType::elem_type InputElemType;
     typedef typename OutputDataType::elem_type ElemType;
   
     FastLSTM();
   
     FastLSTM(const FastLSTM& layer);
   
     FastLSTM(FastLSTM&& layer);
   
     FastLSTM& operator=(const FastLSTM& layer);
   
     FastLSTM& operator=(FastLSTM&& layer);
   
     FastLSTM(const size_t inSize,
              const size_t outSize,
              const size_t rho = std::numeric_limits<size_t>::max());
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output);
   
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
       return 4 * outSize * inSize + 4 * outSize + 4 * outSize * outSize;
     }
   
     size_t InputShape() const
     {
       return inSize;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     template<typename InputType, typename OutputType>
     void FastSigmoid(const InputType& input, OutputType& sigmoids)
     {
       for (size_t i = 0; i < input.n_elem; ++i)
         sigmoids(i) = FastSigmoid(input(i));
     }
   
     ElemType FastSigmoid(const InputElemType data)
     {
       ElemType x = 0.5 * data;
       ElemType z;
       if (x >= 0)
       {
         if (x < 1.7)
           z = (1.5 * x / (1 + x));
         else if (x < 3)
           z = (0.935409070603099 + 0.0458812946797165 * (x - 1.7));
         else
           z = 0.99505475368673;
       }
       else
       {
         ElemType xx = -x;
         if (xx < 1.7)
           z = -(1.5 * xx / (1 + xx));
         else if (xx < 3)
           z = -(0.935409070603099 + 0.0458812946797165 * (xx - 1.7));
         else
           z = -0.99505475368673;
       }
   
       return 0.5 * (z + 1.0);
     }
   
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
   
     OutputDataType output2GateWeight;
   
     OutputDataType input2GateWeight;
   
     OutputDataType input2GateBias;
   
     OutputDataType gate;
   
     OutputDataType gateActivation;
   
     OutputDataType stateActivation;
   
     OutputDataType cell;
   
     OutputDataType cellActivation;
   
     OutputDataType forgetGateError;
   
     OutputDataType prevError;
   
     OutputDataType outParameter;
   
     size_t rhoSize;
   
     size_t bpttSteps;
   }; // class FastLSTM
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "fast_lstm_impl.hpp"
   
   #endif
