
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder.cpp:

Program Listing for File sparse_autoencoder.cpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_sparse_autoencoder.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_autoencoder/sparse_autoencoder.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "sparse_autoencoder.hpp"
   
   namespace mlpack {
   namespace nn {
   
   void SparseAutoencoder::GetNewFeatures(arma::mat& data,
                                          arma::mat& features)
   {
     const size_t l1 = hiddenSize;
     const size_t l2 = visibleSize;
   
     Sigmoid(parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
         arma::repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols),
         features);
   }
   
   } // namespace nn
   } // namespace mlpack
