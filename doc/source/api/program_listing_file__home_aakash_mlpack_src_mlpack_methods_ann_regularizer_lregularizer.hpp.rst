
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_lregularizer.hpp:

Program Listing for File lregularizer.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_lregularizer.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/regularizer/lregularizer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LREGULARIZER_HPP
   #define MLPACK_METHODS_ANN_LREGULARIZER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann {
   
   template<int TPower>
   class LRegularizer
   {
    public:
     LRegularizer(double factor = 1.0);
   
     template<typename MatType>
     void Evaluate(const MatType& weight, MatType& gradient);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     static const int Power = TPower;
   
     double factor;
   };
   
   // Convenience typedefs.
   typedef LRegularizer<1> L1Regularizer;
   
   typedef LRegularizer<2> L2Regularizer;
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "lregularizer_impl.hpp"
   
   #endif
