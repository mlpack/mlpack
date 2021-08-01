
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_lregularizer_impl.hpp:

Program Listing for File lregularizer_impl.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_lregularizer_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/regularizer/lregularizer_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LREGULARIZER_IMPL_HPP
   #define MLPACK_METHODS_ANN_LREGULARIZER_IMPL_HPP
   
   // In case it hasn't been included.
   #include "lregularizer.hpp"
   
   namespace mlpack {
   namespace ann {
   
   template<int Power>
   LRegularizer<Power>::LRegularizer(double factor) :
       factor(factor)
   {
     // Nothing to do here
   }
   
   // Unspecialized implementation. This should almost never be used...
   template<int Power>
   template<typename MatType>
   void LRegularizer<Power>::Evaluate(const MatType& weight, MatType& gradient)
   {
     gradient += arma::vectorise(arma::pow(weight, Power - 1) * Power * factor);
   }
   
   // L1-Regularizer specializations.
   template<>
   template<typename MatType>
   void LRegularizer<1>::Evaluate(const MatType& weight, MatType& gradient)
   {
     gradient += arma::vectorise(factor * weight / arma::abs(weight));
   }
   
   // L2-Regularizer specializations.
   template<>
   template<typename MatType>
   void LRegularizer<2>::Evaluate(const MatType& weight, MatType& gradient)
   {
     gradient += arma::vectorise(2 * factor * weight);
   }
   
   template<int Power>
   template<typename Archive>
   void LRegularizer<Power>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     ar(CEREAL_NVP(factor));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
