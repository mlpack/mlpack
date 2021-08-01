
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout_impl.hpp:

Program Listing for File alpha_dropout_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_alpha_dropout_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/alpha_dropout_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "alpha_dropout.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   AlphaDropout<InputDataType, OutputDataType>::AlphaDropout(
       const double ratio,
       const double alphaDash) :
       ratio(ratio),
       alphaDash(alphaDash),
       deterministic(false)
   {
     Ratio(ratio);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void AlphaDropout<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     // The dropout mask will not be multiplied in the deterministic mode
     // (during testing).
     if (deterministic)
     {
       output = input;
     }
     else
     {
       // Set values to alphaDash with probability ratio.  Then apply affine
       // transformation so as to keep mean and variance of outputs to their
       // original values.
       mask = arma::randu< arma::Mat<eT> >(input.n_rows, input.n_cols);
       mask.transform( [&](double val) { return (val > ratio); } );
       output = (input % mask + alphaDash * (1 - mask)) * a + b;
     }
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void AlphaDropout<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
   {
     g = gy % mask * a;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename Archive>
   void AlphaDropout<InputDataType, OutputDataType>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(ratio));
     ar(CEREAL_NVP(alphaDash));
     ar(CEREAL_NVP(a));
     ar(CEREAL_NVP(b));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
