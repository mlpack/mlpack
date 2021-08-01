
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_maximal_inputs.cpp:

Program Listing for File maximal_inputs.cpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_sparse_autoencoder_maximal_inputs.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/sparse_autoencoder/maximal_inputs.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "maximal_inputs.hpp"
   
   namespace mlpack {
   namespace nn {
   
   void MaximalInputs(const arma::mat& parameters, arma::mat& output)
   {
     arma::mat paramTemp(parameters.submat(0, 0, (parameters.n_rows - 1) / 2 - 1,
                                           parameters.n_cols - 2).t());
     double const mean = arma::mean(arma::mean(paramTemp));
     paramTemp -= mean;
   
     NormalizeColByMax(paramTemp, output);
   }
   
   void NormalizeColByMax(const arma::mat &input,
                          arma::mat &output)
   {
     output.set_size(input.n_rows, input.n_cols);
     for (arma::uword i = 0; i != input.n_cols; ++i)
     {
       const double max = arma::max(arma::abs(input.col(i)));
       if (max != 0.0)
       {
         output.col(i) = input.col(i) / max;
       }
       else
       {
         output.col(i) = input.col(i);
       }
     }
   }
   
   } // namespace nn
   } // namespace mlpack
