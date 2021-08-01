
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_bias_svd_bias_svd_impl.hpp:

Program Listing for File bias_svd_impl.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_bias_svd_bias_svd_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/bias_svd/bias_svd_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_BIAS_SVD_BIAS_SVD_IMPL_HPP
   #define MLPACK_METHODS_BIAS_SVD_BIAS_SVD_IMPL_HPP
   
   namespace mlpack {
   namespace svd {
   
   template<typename OptimizerType>
   BiasSVD<OptimizerType>::BiasSVD(const size_t iterations,
                                   const double alpha,
                                   const double lambda) :
       iterations(iterations),
       alpha(alpha),
       lambda(lambda)
   {
     // Nothing to do.
   }
   
   template<typename OptimizerType>
   void BiasSVD<OptimizerType>::Apply(const arma::mat& data,
                                      const size_t rank,
                                      arma::mat& u,
                                      arma::mat& v,
                                      arma::vec& p,
                                      arma::vec& q)
   {
     // batchSize is 1 in our implementation of Bias SVD.
     // batchSize other than 1 has not been supported yet.
     const int batchSize = 1;
     Log::Warn << "The batch size for optimizing BiasSVD is 1."
         << std::endl;
   
     // Make the optimizer object using a BiasSVDFunction object.
     BiasSVDFunction<arma::mat> biasSVDFunc(data, rank, lambda);
     ens::StandardSGD optimizer(alpha, batchSize,
         iterations * data.n_cols);
   
     // Get optimized parameters.
     arma::mat parameters = biasSVDFunc.GetInitialPoint();
     optimizer.Optimize(biasSVDFunc, parameters);
   
     // Constants for extracting user and item matrices.
     const size_t numUsers = max(data.row(0)) + 1;
     const size_t numItems = max(data.row(1)) + 1;
   
     // Extract user and item matrices, user and item bias from the optimized
     // parameters.
     u = parameters.submat(0, numUsers, rank - 1, numUsers + numItems - 1).t();
     v = parameters.submat(0, 0, rank - 1, numUsers - 1);
     p = parameters.row(rank).subvec(numUsers, numUsers + numItems - 1).t();
     q = parameters.row(rank).subvec(0, numUsers - 1).t();
   }
   
   } // namespace svd
   } // namespace mlpack
   
   #endif
