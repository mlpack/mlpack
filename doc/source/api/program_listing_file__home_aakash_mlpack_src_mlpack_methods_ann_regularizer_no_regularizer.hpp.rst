
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_no_regularizer.hpp:

Program Listing for File no_regularizer.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_regularizer_no_regularizer.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/regularizer/no_regularizer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_NO_REGULARIZER_HPP
   #define MLPACK_METHODS_ANN_NO_REGULARIZER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace ann {
   
   class NoRegularizer
   {
    public:
     NoRegularizer()
     {
       // Nothing to do here.
     };
   
     template<typename MatType>
     void Evaluate(const MatType& /* weight */, MatType& /* gradient */)
     {
       // Nothing to do here.
     }
   };
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
