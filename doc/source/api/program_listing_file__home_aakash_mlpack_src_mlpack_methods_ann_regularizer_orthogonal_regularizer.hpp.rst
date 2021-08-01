
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_orthogonal_regularizer.hpp:

Program Listing for File orthogonal_regularizer.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_orthogonal_regularizer.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/regularizer/orthogonal_regularizer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_HPP
   #define MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class OrthogonalRegularizer
   {
    public:
     OrthogonalRegularizer(double factor = 1.0);
   
     template<typename MatType>
     void Evaluate(const MatType& weight, MatType& gradient);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
     double factor;
   };
   
   } // namespace ann
   } // namespace mlpack
   
   // Include implementation.
   #include "orthogonal_regularizer_impl.hpp"
   
   #endif
