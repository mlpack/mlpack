
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_custom_layer.hpp:

Program Listing for File custom_layer.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_custom_layer.hpp>` (``/home/aakash/mlpack/src/mlpack/tests/custom_layer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_TESTS_CUSTOM_LAYER_HPP
   #define MLPACK_TESTS_CUSTOM_LAYER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   
   
   namespace mlpack {
   namespace ann {
     template <
         class ActivationFunction = LogisticFunction,
         typename InputDataType = arma::mat,
         typename OutputDataType = arma::mat
     >
     using CustomLayer = BaseLayer<
         ActivationFunction, InputDataType, OutputDataType>;
   
   } // namespace ann
   } // namespace mlpack
   
   #endif // MLPACK_TESTS_CUSTOM_LAYER_HPP
