
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_maximal_inputs.hpp:

Program Listing for File maximal_inputs.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_maximal_inputs.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_autoencoder/maximal_inputs.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP
   #define MLPACK_METHODS_NN_MAXIMAL_INPUTS_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace nn {
   
   void MaximalInputs(const arma::mat& parameters, arma::mat& output);
   
   void NormalizeColByMax(const arma::mat& input, arma::mat& output);
   
   } // namespace nn
   } // namespace mlpack
   
   #endif
