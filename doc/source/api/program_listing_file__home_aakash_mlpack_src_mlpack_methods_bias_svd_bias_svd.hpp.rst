
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_bias_svd_bias_svd.hpp:

Program Listing for File bias_svd.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_bias_svd_bias_svd.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/bias_svd/bias_svd.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_BIAS_SVD_BIAS_SVD_HPP
   #define MLPACK_METHODS_BIAS_SVD_BIAS_SVD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <ensmallen.hpp>
   #include <mlpack/methods/cf/cf.hpp>
   
   #include "bias_svd_function.hpp"
   
   namespace mlpack {
   namespace svd {
   
   template<typename OptimizerType = ens::StandardSGD>
   class BiasSVD
   {
    public:
     BiasSVD(const size_t iterations = 10,
             const double alpha = 0.02,
             const double lambda = 0.05);
   
     void Apply(const arma::mat& data,
                const size_t rank,
                arma::mat& u,
                arma::mat& v,
                arma::vec& p,
                arma::vec& q);
   
    private:
     size_t iterations;
     double alpha;
     double lambda;
   };
   
   } // namespace svd
   } // namespace mlpack
   
   // Include implementation.
   #include "bias_svd_impl.hpp"
   
   #endif
