
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_isrlu_impl.hpp:

Program Listing for File isrlu_impl.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_isrlu_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/isrlu_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ISRLU_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_ISRLU_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "isrlu.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   ISRLU<InputDataType, OutputDataType>::ISRLU(const double alpha) :
       alpha(alpha)
   {}
   
   template<typename InputDataType, typename OutputDataType>
   template<typename InputType, typename OutputType>
   void ISRLU<InputDataType, OutputDataType>::Forward(
       const InputType& input, OutputType& output)
   {
     output = arma::ones<OutputDataType>(arma::size(input));
     for (size_t i = 0; i < input.n_elem; ++i)
     {
       output(i) = (input(i) >= 0) ? input(i) : input(i) *
           (1 / std::sqrt(1 + alpha * (input(i) * input(i))));
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename DataType>
   void ISRLU<InputDataType, OutputDataType>::Backward(
       const DataType& input, const DataType& gy, DataType& g)
   {
     derivative.set_size(arma::size(input));
     for (size_t i = 0; i < input.n_elem; ++i)
     {
       derivative(i) = (input(i) >= 0) ? 1 :
           std::pow(1 / std::sqrt(1 + alpha * input(i) * input(i)), 3);
     }
     g = gy % derivative;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void ISRLU<InputDataType, OutputDataType>::serialize(
       Archive& ar,
       const uint32_t /* version */)
   {
     ar(CEREAL_NVP(alpha));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
