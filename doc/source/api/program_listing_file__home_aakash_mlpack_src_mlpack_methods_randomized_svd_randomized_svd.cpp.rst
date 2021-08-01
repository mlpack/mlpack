
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_randomized_svd_randomized_svd.cpp:

Program Listing for File randomized_svd.cpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_randomized_svd_randomized_svd.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/randomized_svd/randomized_svd.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "randomized_svd.hpp"
   
   namespace mlpack {
   namespace svd {
   
   RandomizedSVD::RandomizedSVD(const arma::mat& data,
                                arma::mat& u,
                                arma::vec& s,
                                arma::mat& v,
                                const size_t iteratedPower,
                                const size_t maxIterations,
                                const size_t rank,
                                const double eps) :
       iteratedPower(iteratedPower),
       maxIterations(maxIterations),
       eps(eps)
   {
     if (rank == 0)
     {
       Apply(data, u, s, v, data.n_rows);
     }
     else
     {
       Apply(data, u, s, v, rank);
     }
   }
   
   RandomizedSVD::RandomizedSVD(const size_t iteratedPower,
                                const size_t maxIterations,
                                const double eps) :
       iteratedPower(iteratedPower),
       maxIterations(maxIterations),
       eps(eps)
   {
     /* Nothing to do here */
   }
   
   
   void RandomizedSVD::Apply(const arma::sp_mat& data,
                             arma::mat& u,
                             arma::vec& s,
                             arma::mat& v,
                             const size_t rank)
   {
     // Center the data into a temporary matrix for sparse matrix.
     arma::sp_mat rowMean = arma::sum(data, 1) / data.n_cols;
   
     Apply(data, u, s, v, rank, rowMean);
   }
   
   void RandomizedSVD::Apply(const arma::mat& data,
                             arma::mat& u,
                             arma::vec& s,
                             arma::mat& v,
                             const size_t rank)
   {
     // Center the data into a temporary matrix.
     arma::mat rowMean = arma::sum(data, 1) / data.n_cols + eps;
   
     Apply(data, u, s, v, rank, rowMean);
   }
   
   } // namespace svd
   } // namespace mlpack
