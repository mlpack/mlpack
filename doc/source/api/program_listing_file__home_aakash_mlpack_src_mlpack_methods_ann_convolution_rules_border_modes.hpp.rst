
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_convolution_rules_border_modes.hpp:

Program Listing for File border_modes.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_convolution_rules_border_modes.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/convolution_rules/border_modes.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_BORDER_MODES_HPP
   #define MLPACK_METHODS_ANN_CONVOLUTION_RULES_BORDER_MODES_HPP
   
   namespace mlpack {
   namespace ann {
   
   /*
    * The FullConvolution class represents the full two-dimensional convolution.
    */
   class FullConvolution { /* Nothing to do here */ };
   
   /*
    * The ValidConvolution represents only those parts of the convolution that are
    * computed without the zero-padded edges.
    */
   class ValidConvolution { /* Nothing to do here */ };
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
