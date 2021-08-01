
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_celu_impl.hpp:

Program Listing for File celu_impl.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_celu_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/celu_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CELU_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_CELU_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "celu.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   CELU<InputDataType, OutputDataType>::CELU(const double alpha) :
       alpha(alpha),
       deterministic(false)
   {
     if (alpha == 0)
     {
       Log::Fatal << "The value of alpha cannot be equal to 0, "
                  << "terminating the program." << std::endl;
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void CELU<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = arma::ones<OutputDataType>(arma::size(input));
     for (size_t i = 0; i < input.n_elem; ++i)
     {
       output(i) = (input(i) >= 0) ? input(i) : alpha *
                   (std::exp(input(i) / alpha) - 1);
     }
   
     if (!deterministic)
     {
       derivative.set_size(arma::size(input));
       for (size_t i = 0; i < input.n_elem; ++i)
       {
         derivative(i) = (input(i) >= 0) ? 1 :
             (output(i) / alpha) + 1;
       }
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void CELU<InputDataType, OutputDataType>::Backward(
       const DataType& /* input */, const DataType& gy, DataType& g)
   {
     g = gy % derivative;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void CELU<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(alpha));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
