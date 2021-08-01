
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_elu_impl.hpp:

Program Listing for File elu_impl.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_elu_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/elu_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ELU_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_ELU_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "elu.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   // This constructor is called for SELU activation function.  The values of
   // alpha and lambda are constant for normalized inputs.
   template<typename InputDataType, typename OutputDataType>
   ELU<InputDataType, OutputDataType>::ELU() :
       alpha(1.6732632423543774),
       lambda(1.0507009873554802),
       deterministic(false)
   {
     // Nothing to do here.
   }
   
   // This constructor is called for ELU activation function.  The value of lambda
   // is fixed and equal to 1.  'alpha' is a hyperparameter.
   template<typename InputDataType, typename OutputDataType>
   ELU<InputDataType, OutputDataType>::ELU(const double alpha) :
       alpha(alpha),
       lambda(1),
       deterministic(false)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void ELU<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = arma::ones<OutputDataType>(arma::size(input));
     for (size_t i = 0; i < input.n_elem; ++i)
     {
       if (input(i) < DBL_MAX)
       {
         output(i) = (input(i) > 0) ? lambda * input(i) : lambda *
             alpha * (std::exp(input(i)) - 1);
       }
     }
   
       if (!deterministic)
       {
         derivative.set_size(arma::size(input));
         for (size_t i = 0; i < input.n_elem; ++i)
         {
           derivative(i) = (input(i) > 0) ? lambda : output(i) +
               lambda * alpha;
         }
       }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void ELU<InputDataType, OutputDataType>::Backward(
       const DataType& /* input */, const DataType& gy, DataType& g)
   {
     g = gy % derivative;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void ELU<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(alpha));
     ar(CEREAL_NVP(lambda));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
